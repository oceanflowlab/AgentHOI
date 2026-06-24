import argparse
import asyncio
import base64
import json
import os
import time
from typing import List, Dict, Any, Set, Optional, Tuple

from openai import AsyncOpenAI
from tqdm import tqdm

from pipe_argu import HOI_Remining_arguments
# --- 辅助函数 ---

def local_image_to_base64(image_path: str) -> Optional[str]:
    """将本地图片文件转换为 Base64 编码的字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"图片文件未找到 (用于 Base64 编码): {image_path}")
        return None
    except Exception as e:
        print(f"读取或编码图片 {image_path} 时出错: {e}")
        return None

def load_task_prompt_from_file(prompt_file_path: str) -> str:
    """从文件加载任务提示。"""
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"从 {prompt_file_path} 加载任务提示时出错: {e}")
        raise # 重新抛出异常，因为这是关键步骤

def load_and_filter_existing_results(
    file_path: str
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    从输出文件加载现有结果。
    只保留状态为 "success" 的条目。
    返回:
        - List[Dict[str, Any]]: 过去成功处理的结果列表。
        - Set[str]: 过去成功处理的文件名集合。
    """
    valid_past_results: List[Dict[str, Any]] = []
    processed_successful_filenames: Set[str] = set()

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"输出文件 '{file_path}' 未找到或为空。将从头开始处理。")
        return valid_past_results, processed_successful_filenames

    print(f"从 '{file_path}' 加载已有结果...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
    except json.JSONDecodeError:
        print(f"'{file_path}' 中的 JSON 解码错误。文件可能已损坏。")
        backup_name = file_path + ".corrupted_backup_" + time.strftime("%Y%m%d%H%M%S")
        try:
            os.rename(file_path, backup_name)
            print(f"已将损坏的文件备份到 '{backup_name}'。")
        except OSError:
            print(f"无法备份损坏的文件 '{file_path}'。")
        print("本次将基于空结果列表开始。")
        return [], set() # 如果文件损坏，则从头开始

    if not isinstance(loaded_data, list):
        print(f"警告: '{file_path}' 的内容不是一个 JSON 列表。本次将基于空结果列表开始。")
        return [], set()

    for item in loaded_data:
        if isinstance(item, dict) and item.get("filename"):
            if item.get("state") == "success":
                valid_past_results.append(item)
                processed_successful_filenames.add(item["filename"])
            # 其他状态 (如 error) 的条目将被忽略，并在需要时重新处理
        else:
            print(f"警告: 在 '{file_path}' 中发现非字典或缺少 'filename' 的条目: {item}")

    print(f"从 '{file_path}' 加载了 {len(valid_past_results)} 个过去的成功结果。"
          f"共找到 {len(processed_successful_filenames)} 个已成功处理的文件名。")
    return valid_past_results, processed_successful_filenames


def save_results_atomically(results: List[Dict[str, Any]], target_file_path: str) -> None:
    """原子性地将结果列表保存到目标文件。"""
    temp_file_path = target_file_path + ".tmp"
    try:
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        os.replace(temp_file_path, target_file_path) # 原子性重命名
    except Exception as e:
        print(f"保存结果到 {target_file_path} 时出错: {e}")
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass # 清理临时文件时忽略错误
        raise

# --- 核心逻辑 ---

def extract_info_from_data_item(data_item: Dict[str, Any]) -> Tuple[str, str]:
    """
    从单个 JSON 数据项中提取并格式化信息。
    这个函数源自原始脚本的 `extract_info` 方法。
    """
    filename = data_item["filename"]
    hops_list = []
    # 使用 .get 提供默认空列表，以防键不存在
    num_objects = len(data_item.get("Object Category", []))

    for i in range(num_objects):
        oc = data_item["Object Category"][i]
        ic = data_item["Interaction Class"][i]
        ic_str = " ".join(ic) if isinstance(ic, list) else str(ic)
        hd = data_item["Human Description"][i]
        od = data_item["Object Description"][i]
        hop = f"OC:{oc};IC:{ic_str};HD:{hd};OD:{od}"
        hops_list.append(hop)

    final_hops_str = "HOPs:{" + "\n".join(hops_list) + "}"
    return filename, final_hops_str

async def analyze_single_image_async(
    client: AsyncOpenAI,
    task_prompt_template: str, # 从文件读取的通用系统提示
    item_data: Dict[str, Any], # 当前图片对应的具体数据项
    image_base_dir: str,
    model_name: str,
    max_tokens: int,
    oc_list_info_prompt: str # 包含 OC 列表的特定信息提示文本
) -> Dict[str, Any]:
    """异步分析单个图片。"""
    filename, final_hops_for_item = extract_info_from_data_item(item_data)
    img_url_path = os.path.join(image_base_dir, filename)

    base64_image = local_image_to_base64(img_url_path)
    if base64_image is None:
        return {
            "filename": filename,
            "analysis": f"错误：无法读取或编码图片文件 {img_url_path}。",
            "state": "error_image_encoding" # 使用 "state" 标记状态
        }

    # 构建完整的 API 用户提示文本
    user_text_prompt = final_hops_for_item + oc_list_info_prompt

    try:
        messages = [
            {"role": "system", "content": task_prompt_template}, # 系统提示
            {"role": "user", "content": [ # 用户输入，包含图片和文本
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": user_text_prompt},
            ]}
        ]
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages, #type: ignore
            max_tokens=max_tokens,
            temperature=0, # 原始脚本使用的 temperature
            logprobs=True
        )
        analysis_content = response.choices[0].message.content
        
        # --- 新增：解析 logprobs ---
        detailed_analysis = []
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            for token_info in response.choices[0].logprobs.content:
                token_str = token_info.token
                logprob = token_info.logprob
                # 将对数概率转换为普通概率 (e^logprob)
                probability = 2.71828**logprob # 使用math.exp(logprob)会更精确
                
                detailed_analysis.append({
                    "token": token_str,
                    "probability": probability
                })
                
                
        return {
            "filename": filename,
            "analysis": analysis_content, # 保留原始的完整句子
            "detailed_analysis": detailed_analysis, # 新增带有概率的详细分析
            "state": "success"
        }
    except Exception as e:
        print(f"处理图片 {filename} 时 API 发生错误: {e}")
        return {
            "filename": filename,
            "analysis": str(e),
            "detailed_analysis": [],
            "state": "error_api_call" # API 调用错误状态
        }

async def process_all_images_async(
    args: argparse.Namespace,
    full_input_data: List[Dict[str, Any]], # 完整的输入数据列表
    task_prompt_template: str,
    oc_list_info_prompt: str,
    output_file: str
):
    """异步处理所有需要分析的图片，并保存结果。"""

    # 加载过去成功的结果，这些结果将作为我们当前结果列表的基础
    # `processed_successful_filenames_from_past` 用于确定哪些新图片需要处理
    current_results_to_save, processed_successful_filenames_from_past = \
        load_and_filter_existing_results(output_file)

    # 从完整的输入数据中筛选出本次需要处理的图片
    # 即那些文件名不在 `processed_successful_filenames_from_past` 中的图片
    items_to_process_this_run = [
        item for item in full_input_data
        if item["filename"] not in processed_successful_filenames_from_past
    ]

    if not items_to_process_this_run:
        print(f"所有 {len(full_input_data)} 张图片均已成功处理过。无需额外操作。")
        # 确保输出文件存在且内容是正确的（即上次成功的列表）
        if not os.path.exists(output_file) and not current_results_to_save:
             save_results_atomically([], output_file) # 如果文件不存在且无历史成功记录，则创建一个空列表文件
        elif current_results_to_save : # 如果有历史成功记录（无论文件是否存在，理论上应该存在），则保存它们
             save_results_atomically(current_results_to_save, output_file)
        return

    print(f"共发现 {len(full_input_data)} 张图片。其中 {len(items_to_process_this_run)} 张图片需要处理（新的或之前失败的）。")
    print(f"已有 {len(processed_successful_filenames_from_past)} 张图片之前已成功处理。")


    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base_url)
    
    # `newly_processed_results_this_session_buffer` 用于暂存当前运行会话中新处理的结果
    newly_processed_results_this_session_buffer: List[Dict[str, Any]] = []
    
    total_to_process_in_this_session = len(items_to_process_this_run)
    successfully_processed_count_this_session = 0
    failed_count_this_session = 0

    try:
        # tqdm 进度条现在反映的是当前会话中需要处理的图片数量
        with tqdm(total=total_to_process_in_this_session, desc="分析图片中") as pbar:
            for i in range(0, total_to_process_in_this_session, args.concurrency):
                batch_items_to_process = items_to_process_this_run[i : i + args.concurrency]
                
                async_tasks = [
                    analyze_single_image_async(
                        client,
                        task_prompt_template,
                        item_data,
                        args.image_base_dir,
                        args.model,
                        args.max_tokens,
                        oc_list_info_prompt
                    )
                    for item_data in batch_items_to_process
                ]
                
                batch_results_from_api = await asyncio.gather(*async_tasks)
                
                newly_processed_results_this_session_buffer.extend(batch_results_from_api)
                pbar.update(len(batch_items_to_process))

                # 更新当前会话的成功和失败计数
                for res in batch_results_from_api:
                    if res.get("state") == "success":
                        successfully_processed_count_this_session += 1
                    else:
                        failed_count_this_session += 1
                
                # 保存点逻辑：达到保存间隔，或者是最后一个批次且缓冲区有内容
                is_last_batch_iteration = (i + args.concurrency >= total_to_process_in_this_session)
                if (len(newly_processed_results_this_session_buffer) >= args.save_interval) or \
                   (is_last_batch_iteration and newly_processed_results_this_session_buffer):
                    
                    # 将当前会话缓冲区中的新结果合并到 `current_results_to_save`
                    # `current_results_to_save` 已包含所有过去的成功结果
                    current_results_to_save.extend(newly_processed_results_this_session_buffer)
                    
                    print(f"\n保存点：本会话新处理 {len(newly_processed_results_this_session_buffer)} 个图片。"
                          f"内存中结果总数: {len(current_results_to_save)}。正在保存到 '{output_file}'...")
                    try:
                        save_results_atomically(current_results_to_save, output_file)
                        print(f"成功保存 {len(current_results_to_save)} 个结果。")
                        newly_processed_results_this_session_buffer.clear() # 保存成功后清空缓冲区
                    except Exception as e:
                        print(f"周期性保存时发生严重错误: {e}。后续保存可能受影响。")
                        # 如果保存失败，缓冲区中的内容会保留，下次保存时会再次尝试
    finally:
        await client.close()
        print("OpenAI 客户端已关闭。")

    # 最终保存，处理在最后一个保存点之后、客户端关闭之前可能仍在缓冲区中的项目
    # (正常情况下，如果最后一个批次的保存成功，此缓冲区应为空)
    if newly_processed_results_this_session_buffer:
        current_results_to_save.extend(newly_processed_results_this_session_buffer)
        newly_processed_results_this_session_buffer.clear()

    print("正在最终确定结果...")
    try:
        save_results_atomically(current_results_to_save, output_file)
        print(f"所有 {len(current_results_to_save)} 个结果已保存到 '{output_file}'。")
    except Exception as e:
        print(f"最终保存时发生严重错误: {e}。")

    print("图片分析流程完成。")
    print(f"  本会话成功处理: {successfully_processed_count_this_session}")
    print(f"  本会话处理失败: {failed_count_this_session}")
    print(f"  输出文件 '{output_file}' 中的总项目数: {len(current_results_to_save)}")


def main():
    args = HOI_Remining_arguments()

    if not args.api_key:
        print("错误：必须提供 OpenAI API 密钥。请设置 OPENAI_API_KEY 环境变量或使用 --api_key 参数。")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir: # 如果 output_file 包含路径
        os.makedirs(output_dir, exist_ok=True)


    # 加载主任务提示 (系统提示)
    task_prompt_template = load_task_prompt_from_file(args.prompt_file)

    # 这是原始脚本中固定的、附加到每个用户提示后的信息部分
    oc_list_info_prompt = (
        "\n Do all the OCs of the HOPs appear in the image? If not, delete those HOPs that OC do not appear. "
        "Are there any obvious objects in the OC list (OC list:[airplane apple backpack banana baseball_bat "
        "baseball_glove bear bed bench bicycle bird boat book bottle bowl broccoli bus cake car carrot cat "
        "cell_phone chair clock couch cow cup dining_table dog donut elephant fire_hydrant fork frisbee "
        "giraffe hair_drier handbag horse hot_dog keyboard kite knife laptop microwave motorcycle mouse "
        "orange oven parking_meter person pizza potted_plant refrigerator remote sandwich scissors sheep "
        "sink skateboard skis snowboard spoon sports_ball stop_sign suitcase surfboard teddy_bear "
        "tennis_racket tie toaster toilet toothbrush traffic_light train truck tv umbrella vase "
        "wine_glass zebra]) that haven't been output in the image? If so, please supplement corresponding HOPs. "
        "If no the two problems, output the original HOPs directly, don't output any other content"
    )

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            full_input_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：输入 JSON 文件未找到: {args.input_file}")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解码输入文件中的 JSON: {args.input_file}")
        return

    if not full_input_data:
        print(f"输入文件 {args.input_file} 为空或不包含任何数据项。正在退出。")
        # 确保如果输出文件不存在，则创建一个空列表文件
        if not os.path.exists(args.output_file):
            save_results_atomically([], args.output_file)
        return

    print(f"开始处理输入文件: {args.input_file} ({len(full_input_data)} 个项目)。")
    print(f"输出将保存到: {args.output_file}")
    print(f"并发数: {args.concurrency}, 保存间隔: 每处理 {args.save_interval} 个新项目。")
    print(f"使用模型: {args.model}, 最大 Token 数: {args.max_tokens}")

    try:
        asyncio.run(process_all_images_async(
            args,
            full_input_data,
            task_prompt_template,
            oc_list_info_prompt,
            args.output_file
        ))
    except Exception as e:
        print(f"异步处理过程中发生未处理的错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()