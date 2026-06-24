import warnings
warnings.filterwarnings('ignore')
import json
import os
import torch
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import gc
import time
import argparse
import multiprocessing as mp
from functools import partial
from pipe_argu import Box_arguments
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from GroundingDINO.outbox_func import load_image, load_model, get_grounding_output

# --- Helper Function ---
def box_cxcywh_to_xyxy(image_wh: Tuple[int, int], box: torch.Tensor) -> torch.Tensor:
    """将归一化的 [cx, cy, w, h] 框转换为绝对坐标 [x1, y1, x2, y2] 框。"""
    w, h = image_wh
    cx, cy, bw, bh = box.unbind(-1)
    x1 = (cx - 0.5 * bw) * w
    y1 = (cy - 0.5 * bh) * h
    x2 = (cx + 0.5 * bw) * w
    y2 = (cy + 0.5 * bh) * h
    return torch.stack((x1, y1, x2, y2), dim=-1)

# --- Model Configuration (Worker Scope) ---
CONFIG_FILE = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
CHECKPOINT_PATH = "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
BOX_THRESHOLD = 0.1  # 从 BoxGenerator 移至此处或作为参数传递
TEXT_THRESHOLD = 0.15 # 从 BoxGenerator 移至此处或作为参数传递

# Global dictionary to store model per worker process
worker_globals = {}

def init_worker(gpu_ids: List[int]):
    """Initializes the worker process: loads the model onto the assigned GPU."""
    global worker_globals
    process_idx = mp.current_process()._identity[0] - 1
    if not gpu_ids:
        raise ValueError("GPU ID list cannot be empty for worker initialization.")
    gpu_id = gpu_ids[process_idx % len(gpu_ids)]
    process_name = mp.current_process().name
    print(f"Initializing {process_name} on GPU {gpu_id}...")
    try:
        torch.cuda.set_device(gpu_id)
        model = load_model(CONFIG_FILE, CHECKPOINT_PATH, cpu_only=False)
        model.eval()
        worker_globals['model'] = model
        worker_globals['gpu_id'] = gpu_id
        print(f"Model loaded successfully on GPU {gpu_id} for {process_name}.")
    except Exception as e:
        print(f"Error initializing worker on GPU {gpu_id} ({process_name}): {e}")
        raise

def get_top_logit_from_detections(
    model,
    image_pil, # Type hint for PIL Image
    image_tensor: torch.Tensor,
    expression_text: str,
    category_text: str,
    interaction:str,
    isHuman: bool,
    gpu_id: int
) -> Optional[List[float]]:
    """
    Helper function to get the box with the highest logit for a given expression.
    This combines logic from the original `get_top_logit`.
    """
    
    prefix = "There is a "
    suffix = " that "
    if isHuman: 
        num_category_chars = len("person")
        full_expression = f"{prefix}person{suffix}{expression_text} {interaction} a {category_text} ."
        start_idx = len(prefix)
        end_idx = start_idx + num_category_chars
    else:
        num_category_chars = len(category_text)
        full_expression = f"{prefix}{category_text}{suffix}{expression_text} {interaction} by a person ."
        start_idx = len(prefix)
        end_idx = start_idx + num_category_chars
    # 修正 caption 和 token_spans 的生成逻辑

    token_spans_for_category = [[[start_idx, end_idx]]]
    if isHuman:
        print(f"GPU {gpu_id}: Getting grounding for category 'person' with expression: '{full_expression}', token_spans: {token_spans_for_category}")

    else:
        print(f"GPU {gpu_id}: Getting grounding for category '{category_text}' with expression: '{full_expression}', token_spans: {token_spans_for_category}")

    image_tensor_on_gpu = image_tensor.to(f'cuda:{gpu_id}')

    with torch.no_grad():
        boxes_filt, pred_phrases, logits = get_grounding_output(
            model, image_tensor_on_gpu, full_expression, BOX_THRESHOLD, TEXT_THRESHOLD,
            cpu_only=False, token_spans=token_spans_for_category
        )

    if boxes_filt is None or logits is None or len(boxes_filt) == 0 or len(logits) == 0:
        # print(f"GPU {gpu_id}: No detections for category '{category_text}' with expression '{expression_text}'")
        return None
    
    if isHuman:
        print(f"GPU {gpu_id}: Raw pred_phrases for 'person': {pred_phrases}, logits: {logits.tolist()}")

    else:
        print(f"GPU {gpu_id}: Raw pred_phrases for '{category_text}': {pred_phrases}, logits: {logits.tolist()}")


    # 筛选与目标类别相关的检测结果
    # GroundingDINO有时会返回与token_spans不完全匹配的短语，但logit可能很高。
    # 原始代码逻辑是直接取 logits.argmax()，这假设了模型输出的第一个最高logit对应我们想要的类别。
    # 一个更稳健的方法是检查 pred_phrases 是否与 category_text 相关。
    
    relevant_indices = []
    if pred_phrases:
        for i, phrase in enumerate(pred_phrases):
            # 如果模型能精确返回由token_span指定的类别名，这是理想情况
            if phrase and category_text.lower() in phrase.lower():
                relevant_indices.append(i)
    
    if not relevant_indices and len(logits) > 0:
        # 如果没有精确匹配的短语，但模型通过token_spans被引导了，
        # 并且原始代码依赖于最高logit，我们回退到该逻辑。
        # 这假设了即使短语不完全匹配，最高logit的框仍然是与指定类别相关的。
        # print(f"GPU {gpu_id}: No exact phrase match for '{category_text}', using highest logit overall.")
        max_logit_index = logits.argmax().item()
        relevant_indices = [max_logit_index]


    if not relevant_indices:
        # print(f"GPU {gpu_id}: Still no relevant detections after filtering for '{category_text}'.")
        return None

    if relevant_indices:
        target_logits = logits[relevant_indices]
        target_boxes = boxes_filt[relevant_indices]
        
        if len(target_logits) > 0:
            max_logit_in_target_idx = target_logits.argmax().item() # 相对于 target_logits 和 target_boxes 的索引
            # best_box_index_in_filt = relevant_indices[max_logit_in_target_idx] # 原始 boxes_filt 中的索引

            # 获取对应的box和logit
            # box_cxcywh = boxes_filt[best_box_index_in_filt]
            box_cxcywh = target_boxes[max_logit_in_target_idx]
            logit_float = target_logits[max_logit_in_target_idx].item()

            size_wh = image_pil.size # W, H
            box_xyxy = box_cxcywh_to_xyxy(size_wh, box_cxcywh.cpu()) # 转换为 [x1,y1,x2,y2]
            
            box_coords_list = box_xyxy.numpy().tolist()
            box_coords_list.append(logit_float) # [x1, y1, x2, y2, logit]
            # print(f"GPU {gpu_id}: Found box for '{category_text}' with logit {logit_float}: {box_coords_list[:4]}")
            return box_coords_list
        else:
            # print(f"GPU {gpu_id}: No logits found for relevant indices of '{category_text}'.")
            return None
    else: # 这种情况理论上不应发生，因为前面已经处理了 relevant_indices 为空的情况
        return None


def process_single_item_worker(item_data: Dict, image_folder: str) -> Optional[Dict]:
    """
    Processes a single image item within a worker process.
    This function integrates the logic from `BoxGenerator.process_single_image`
    and `BoxGenerator.get_top_logit`.
    """
    global worker_globals
    model = worker_globals.get('model')
    gpu_id = worker_globals.get('gpu_id', -1)

    if model is None or gpu_id == -1:
        print(f"Error: Worker (PID {os.getpid()}) not properly initialized. Skipping item.")
        return None # Or return item with error status

    filename = item_data.get('filename')
    if not filename:
        print(f"Error: Item is missing 'filename'. Skipping.")
        return None

    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        print(f"Warning: Image {filename} not found at {image_path} on GPU {gpu_id}. Skipping.")
        # 返回原始项目，但标记错误或box为空
        result_shell = {
            'filename': filename,
            'Object Category': item_data.get('Object Category', []),
            'Interaction Class': item_data.get('Interaction Class', []),
            'Human Description': item_data.get('Human Description', []),
            'Object Description': item_data.get('Object Description', []),
            'Box': [], # 将保持为空
            'orig_size': item_data.get('orig_size', []),
            'error': f"Image not found: {image_path}"
        }
        return result_shell
    
    # print(f"Processing {filename} on GPU {gpu_id} (PID {os.getpid()})")

    # Prepare the result structure based on the first code snippet
    processed_item_result = {
        'filename': filename,
        'Object Category': item_data.get('Object Category', []),
        'Object Logits': item_data.get('Object Logits', []),
        'IC Logits': item_data.get('IC Logits', []),
        'Interaction Class': item_data.get('Interaction Class', []),
        'Human Description': item_data.get('Human Description', []),
        'Object Description': item_data.get('Object Description', []),
        'Box': [], # This will be populated
        'orig_size': item_data.get('orig_size', []) # Assuming this is [W, H] or similar
    }

    try:
        image_pil, image_tensor = load_image(image_path) # Loads once per item

        num_descriptions = len(item_data.get('Human Description', []))
        
        for i in range(num_descriptions):
            torch.cuda.empty_cache() # As per original logic

            human_desc = item_data['Human Description'][i]
            obj_desc = item_data['Object Description'][i]
            obj_category = item_data['Object Category'][i]
            interaction_class = item_data['Interaction Class'][i]
            ic = interaction_class[0] # Get the first word of the interaction class
            
            # Get human box
            human_box_with_logit = get_top_logit_from_detections(
                model, image_pil, image_tensor,
                expression_text=human_desc.strip(), # Original: human_desc + " ."
                category_text=obj_category,
                interaction=ic,
                isHuman=True,
                gpu_id=gpu_id
            )

            # Get object box
            obj_box_with_logit = get_top_logit_from_detections(
                model, image_pil, image_tensor,
                expression_text=obj_desc.strip(), # Original: obj_desc + " ."
                category_text=obj_category,
                interaction=ic,
                isHuman=False,
                gpu_id=gpu_id
            )

            box_info = {
                'humanBox': human_box_with_logit if human_box_with_logit else [], # Ensure it's a list
                'objBox': obj_box_with_logit if obj_box_with_logit else [],     # Ensure it's a list
                'Interaction Class': interaction_class
            }
            processed_item_result['Box'].append(box_info)
        
        del image_pil, image_tensor # Clean up after processing all descriptions for this image

    except Exception as e:
        print(f"Error processing {filename} on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        # Optionally, mark this item as failed in the result
        processed_item_result['error'] = str(e)
        # Ensure 'Box' is still a list even if processing fails midway
        if 'Box' not in processed_item_result or not isinstance(processed_item_result['Box'], list):
             processed_item_result['Box'] = []


    return processed_item_result


def main():
    args = Box_arguments()

    available_gpus = list(range(torch.cuda.device_count()))
    
    active_gpu_ids = available_gpus[:args.num_workers] if args.num_workers > 0 else []


    print(f"Loading input data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        all_input_data = json.load(f)

    checkpoint_file = args.output_file.replace('.json', '_checkpoint.json')
    processed_results = []
    processed_filenames = set()

    if os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            try:
                processed_results = json.load(f)
                # Ensure processed_results is a list
                if not isinstance(processed_results, list):
                    print(f"Warning: Checkpoint file {checkpoint_file} does not contain a valid list. Starting fresh.")
                    processed_results = []
                
                for item in processed_results:
                    if isinstance(item, dict) and 'filename' in item:
                        processed_filenames.add(item['filename'])
            except json.JSONDecodeError:
                print(f"Warning: Checkpoint file {checkpoint_file} is corrupted. Starting fresh.")
                processed_results = [] # Reset if corrupted
        print(f"Loaded {len(processed_results)} results from checkpoint. {len(processed_filenames)} unique filenames processed.")

    data_to_process = [item for item in all_input_data if isinstance(item, dict) and item.get('filename') not in processed_filenames]
    
    original_total = len(all_input_data)
    remaining_to_process_count = len(data_to_process)
    print(f"Total items in input: {original_total}. Already processed: {len(processed_filenames)}. Remaining to process: {remaining_to_process_count}.")

    if not data_to_process:
        print("All items seem to be processed according to the checkpoint.")
        if os.path.exists(checkpoint_file) and not os.path.exists(args.output_file):
            try:
                os.rename(checkpoint_file, args.output_file)
                print(f"Renamed checkpoint to final output: {args.output_file}")
            except OSError as e:
                print(f"Error renaming checkpoint file: {e}. Final results might be in {checkpoint_file}")
        elif not os.path.exists(args.output_file) and processed_results: # If output doesn't exist but we have results from checkpoint
             print(f"Saving final results from checkpoint to {args.output_file}...")
             with open(args.output_file, 'w', encoding='utf-8') as f:
                 json.dump(processed_results, f, indent=2, ensure_ascii=False)
             print("Final results saved.")
        return

    if args.num_workers == 0: # Or if no GPUs and we decide to run sequentially on CPU
        print("Running in single-process mode (num_workers is 0 or no GPUs for workers).")
        print("Single-process mode for this script structure is not fully set up if models require GPU. Exiting if GPUs were expected.")
        if len(available_gpus) > 0 : # If GPUs were available but num_workers forced to 0
             pass # Allow to proceed if user explicitly set num_workers = 0 for some reason
        else: # No GPUs and num_workers is 0
             print("Cannot proceed: GPU-dependent components and no GPUs available for workers, and single-CPU path not fully implemented.")
             return
         
    mp.set_start_method('spawn', force=True)
    
    # Create a partial function with fixed arguments for the worker
    # The `image_folder` argument is fixed here
    worker_func_with_args = partial(process_single_item_worker, image_folder=args.image_folder)
    
    newly_processed_count_this_run = 0
    start_time = time.time()

    print(f"Starting {args.num_workers} worker processes with GPUs: {active_gpu_ids}...")
    try:
        with mp.Pool(processes=args.num_workers, initializer=init_worker, initargs=(active_gpu_ids,)) as pool:
            # Using imap_unordered for potentially better performance as results are processed as they complete
            results_iterator = pool.imap_unordered(worker_func_with_args, data_to_process)
            
            for result in tqdm(results_iterator, total=remaining_to_process_count, desc="Processing images"):
                if result: # Worker function returns None on some errors
                    processed_results.append(result)
                    newly_processed_count_this_run += 1

                    if newly_processed_count_this_run > 0 and \
                       (newly_processed_count_this_run % args.checkpoint_interval == 0 or \
                        newly_processed_count_this_run == remaining_to_process_count): # Also save at the very end of new items
                        
                        print(f"\nSaving checkpoint ({len(processed_results)} total items)...")
                        with open(checkpoint_file, 'w', encoding='utf-8') as f_check:
                            json.dump(processed_results, f_check, indent=2, ensure_ascii=False)
                        
                        # Optional: save intermediate full output file as well, like in original script
                        with open(args.output_file, 'w', encoding='utf-8') as f_out:
                            json.dump(processed_results, f_out, indent=2, ensure_ascii=False)

                        print(f"Checkpoint and results saved after processing {newly_processed_count_this_run} new items (total {len(processed_results)}).")
                        
                        # Memory cleanup (less critical here as it's in worker, but can be added if main process memory grows)
                        gc.collect()
                        # No need to torch.cuda.empty_cache() in main process for worker GPUs generally
                        # If there was any GPU activity in the main process, it could be done.
    
    except Exception as e:
        print(f"\nAn error occurred during multiprocessing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Pool is closed automatically by `with` statement
        print("Worker pool processing finished or an error occurred.")

    # --- Final Save ---
    print("\nProcessing complete for this run.")
    print(f"Items processed in this run: {newly_processed_count_this_run}")
    print(f"Total accumulated results: {len(processed_results)}")

    print(f"Saving final aggregated results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    print("Final results saved successfully.")

    # Cleanup checkpoint file after successful final save
    if os.path.exists(checkpoint_file):
        try:
            # Decide whether to remove or keep. Original second script kept it. First script implies removal implicitly by overwriting output.
            # For safety, let's print a message to manually remove or keep.
            os.remove(checkpoint_file)
            print(f"Checkpoint file '{checkpoint_file}' still exists. You may want to remove it manually if the final output is complete.")
        except OSError as e:
            print(f"Warning: Could not remove checkpoint file '{checkpoint_file}'. Error: {e}")

    end_time = time.time()
    print(f"Total execution time for this run: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    # This check is important for multiprocessing with 'spawn' or 'forkserver' start methods
    # to prevent issues with re-importing and re-executing the main script in child processes.
    main()