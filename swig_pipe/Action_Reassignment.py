import argparse
import asyncio
import base64
import json
import os
import time
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict
from copy import deepcopy

from openai import AsyncOpenAI
from tqdm import tqdm

from pipe_argu import Action_Reassignment_arguments # 保留您的导入
from swig_v1_categories import SWIG_CATEGORIES, SWIG_ACTIONS, SWIG_INTERACTIONS

# --- 辅助函数 ---
def local_image_to_base64(image_path: str) -> Optional[str]:
    """将本地图片文件转换为 Base64 编码的字符串。"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_task_prompt_from_file(prompt_file_path: str) -> str:
    """从文件加载任务提示。"""

    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_and_filter_results_with_state(
    file_path: str,
    expected_state: str = "success"
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    从输出文件加载现有结果，并基于 'state' 字段进行过滤。
    返回:
        - List[Dict[str, Any]]: 过去符合 `expected_state` 的结果列表。
        - Set[str]: 过去符合 `expected_state` 的文件名集合。
    """
    valid_past_results: List[Dict[str, Any]] = []
    processed_target_state_filenames: Set[str] = set()

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return valid_past_results, processed_target_state_filenames

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
        return [], set()
    
    # 添加对 loaded_data 是否为列表的检查
    if not isinstance(loaded_data, list):
        print(f"警告: '{file_path}' 的内容不是一个JSON列表。文件可能已损坏或格式不正确。")
        # 可以选择在此处尝试备份并返回空，或者根据具体需求处理
        # 为安全起见，返回空，避免后续处理非列表数据时出错
        return [], set()


    for item in loaded_data:
        if isinstance(item, dict) and item.get("filename"):
            if item.get("state") == expected_state:
                valid_past_results.append(item)
                processed_target_state_filenames.add(item["filename"])
        else:
            print(f"警告: 在 '{file_path}' 中发现非字典或缺少 'filename' 的条目: {item}")
    
    return valid_past_results, processed_target_state_filenames

def save_results_atomically(results: List[Dict[str, Any]], target_file_path: str) -> None:
    """原子性地将结果列表保存到目标文件。"""
    temp_file_path = target_file_path + ".tmp"
    try:
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        os.replace(temp_file_path, target_file_path)
    except Exception as e:
        print(f"保存结果到 {target_file_path} 时出错: {e}")
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass
        raise # 将异常重新抛出，以便上层可以感知到保存失败

# --- 核心逻辑与数据处理 ---
def build_object_to_actions_map() -> defaultdict:
    """构建对象到其可能动作列表的映射。"""
    object_to_actions = defaultdict(list)
   
    id2obj = {}
    for obj in SWIG_CATEGORIES:
        id2obj[obj["id"]] = obj["name"]

    id2act = {}
    for act in SWIG_ACTIONS:
        id2act[act["id"]] = act["name"]

    for interaction in SWIG_INTERACTIONS:
        if interaction['evaluation'] == 1:
            obj = id2obj[interaction["object_id"]]
            act = id2act[interaction["action_id"]]
            object_to_actions[obj].append(act)

    
    return object_to_actions

def extract_prompt_info_for_item(
    item_data: Dict[str, Any],
    object_to_actions: defaultdict
) -> Tuple[str, str, List[str]]:
    """
    为单个数据项提取用于构建 API 提示的信息。
    返回: (文件名, API提示信息文本, 该项所有对象可能的交互类别列表 (用于后续验证))
    """
    filename = item_data["filename"]
    num_objects = len(item_data.get("Object Category", []))

    api_prompt_text = ""
    potential_ics_for_validation = [] 

    if num_objects == 0:
        return filename, api_prompt_text, potential_ics_for_validation

    for i in range(num_objects):
        oc = item_data["Object Category"][i]
        hd = item_data["Human Description"][i]
        od = item_data["Object Description"][i]

        hop_base = f"OC:{oc};IC: ;HD:{hd};OD:{od}"
        
        possible_actions_for_oc = [
            action for action in object_to_actions.get(oc, [])
        ]
        ic_list_str = " ".join(possible_actions_for_oc)
        potential_ics_for_validation.append(ic_list_str) 

        hop_prompt_part = f"HOP:{{{hop_base}}}, You now select one or more appropriate ICs from {{IC list:[{ic_list_str}]}}\n"
        api_prompt_text += hop_prompt_part
    
    return filename, api_prompt_text, potential_ics_for_validation

async def call_openai_api_async(
    client: AsyncOpenAI,
    system_task_prompt: str,
    image_path: str,
    user_prompt_text_for_item: str, 
    model_name: str,
    max_tokens: int
) -> Tuple[Optional[str], str]:
    """
    异步调用 OpenAI API。
    返回: (API响应内容字符串 或 None, 状态字符串 "success" 或 "error_...")
    """
    base64_image = local_image_to_base64(image_path)
    if base64_image is None:
        return None, [], "error_image_encoding"

    final_user_prompt = user_prompt_text_for_item + "\nNow reply appropriate interactions"
    try:
        messages = [
            {"role": "system", "content": system_task_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": final_user_prompt},
            ]}
        ]
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages, #type: ignore
            max_tokens=max_tokens,
            logprobs=True,
            temperature=0
        )
        
        
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
        # --- 解析结束 ---
        
        return response.choices[0].message.content, detailed_analysis, "success"
    except Exception as e:
        print(f"处理图片 {os.path.basename(image_path)} 时 API 发生错误: {e}")
        return str(e), [], "error_api_call"

def update_item_with_gpt_result(
    original_item_data: Dict[str, Any],
    gpt_result_content: Optional[str], 
    potential_ics_lists_for_validation: List[str] 
) -> Dict[str, Any]:
    """
    使用 GPT 返回的结果更新单个数据项。
    此函数源自原始脚本的 `process_single_result`。
    返回更新后的数据项 (包含新的 "state" 字段)。
    """
    updated_item = deepcopy(original_item_data)

    if gpt_result_content is None : 
        updated_item["state"] = "error_preprocessing" 
        return updated_item
        
    if not gpt_result_content.strip(): 
        print(f"警告: 文件 {updated_item['filename']} 的 GPT 结果为空。")
        updated_item["state"] = "error_gpt_empty_result"
        return updated_item

    hop_results_str_list = gpt_result_content.split(';')
    hop_results_str_list = [r.strip() for r in hop_results_str_list if r.strip()]

    if len(hop_results_str_list) != len(updated_item.get("Interaction Class", [])):
        print(f"错误: 文件 {updated_item['filename']} 的 GPT 结果数量 ({len(hop_results_str_list)}) "
              f"与 HOP 数量 ({len(updated_item.get('Interaction Class', []))}) 不匹配。GPT原始结果: '{gpt_result_content}'")
        updated_item["state"] = "error_gpt_result_mismatch"
        return updated_item
    
    all_hops_processed_successfully = True
    for i, hop_gpt_actions_str in enumerate(hop_results_str_list):
        if i >= len(updated_item["Interaction Class"]): 
            all_hops_processed_successfully = False
            break
        
        gpt_selected_actions = hop_gpt_actions_str.split()
        
        current_hop_potential_ics_str = potential_ics_lists_for_validation[i]
        validated_actions = [
            act for act in gpt_selected_actions
            if act in current_hop_potential_ics_str 
        ]
        
        if not validated_actions:
            print(f"警告: 文件 {updated_item['filename']}, HOP {i+1}, GPT 选择的动作均不在原始可选列表中。GPT动作: '{gpt_selected_actions}', 可选列表: '{current_hop_potential_ics_str}'. 将使用可选列表中的第一个动作作为默认值。")
            first_potential_action = current_hop_potential_ics_str.split()[0] if current_hop_potential_ics_str.strip() else "unknown_interaction" 
            validated_actions = [first_potential_action]

        updated_item["Interaction Class"][i] = validated_actions

    updated_item["state"] = "success" if all_hops_processed_successfully else "error_processing_some_hops" 
    return updated_item

# +++ 新增的辅助异步函数 +++
async def _get_and_package_api_result(
    client_obj: AsyncOpenAI, 
    sys_prompt: str, 
    image_file_path: str, 
    user_prompt: str, 
    model: str, 
    tokens: int,
    # 需要一起打包的其他参数
    original_filename: str,
    original_potential_ics: List[str],
    original_item_data_dict: Dict[str, Any] # 修改变量名以避免与关键字冲突
) -> Tuple[str, Optional[str], str, List[str], Dict[str, Any]]:
    """辅助函数：调用API并将其结果与原始项目数据打包。"""
    gpt_response_content, detailed_analysis, api_status = await call_openai_api_async(
        client_obj, sys_prompt, image_file_path, user_prompt, model, tokens
    )
    return (
        original_filename,
        gpt_response_content,
        detailed_analysis,
        api_status,
        original_potential_ics,
        original_item_data_dict # 返回修改后的变量名
    )

async def process_all_data_async(args: argparse.Namespace):
    """主处理函数：异步处理所有数据。"""

    print("初始化资源...")
    hico_object_to_actions = build_object_to_actions_map()
    system_task_prompt = load_task_prompt_from_file(args.prompt_file)
    
    print(f"加载输入数据从: {args.input_file}") # 在您的代码中是 args.input_file
    try: # 添加 try-except 处理文件加载
        with open(args.input_file, 'r', encoding='utf-8') as f:
            full_input_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：输入 JSON 文件未找到: {args.input_file}")
        return
    except json.JSONDecodeError:
        print(f"错误：无法解码输入文件中的 JSON: {args.input_file}")
        return
    except Exception as e:
        print(f"加载输入文件 {args.input_file} 时发生未知错误: {e}")
        return


    print("加载已处理的结果...")
    main_output_past_successful_items, main_output_successful_filenames = \
        load_and_filter_results_with_state(args.output_main_file, "success")
    
    ic_results_past_successful_items, _ = \
        load_and_filter_results_with_state(args.output_ic_result_file, "success")

    print(f"主输出文件找到 {len(main_output_successful_filenames)} 个已成功处理的项。")
    
    items_to_process_this_run = [
        item for item in full_input_data
        if item["filename"] not in main_output_successful_filenames
    ]

    if not items_to_process_this_run:
        print(f"所有 {len(full_input_data)} 个项目均已在主输出文件中标记为成功处理。无需额外操作。")
        if not os.path.exists(args.output_main_file) and not main_output_past_successful_items:
            save_results_atomically([], args.output_main_file)
        elif main_output_past_successful_items: # 即使文件存在，也重新保存以确保原子性和格式正确
            save_results_atomically(main_output_past_successful_items, args.output_main_file)
        
        if not os.path.exists(args.output_ic_result_file) and not ic_results_past_successful_items:
            save_results_atomically([], args.output_ic_result_file)
        elif ic_results_past_successful_items: # 同上，重新保存
            save_results_atomically(ic_results_past_successful_items, args.output_ic_result_file)
        return

    print(f"总共 {len(full_input_data)} 个项目。本次将处理 {len(items_to_process_this_run)} 个新项目或之前未成功的项目。")

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base_url)
    
    current_main_output_data = list(main_output_past_successful_items) 
    current_ic_results_data = list(ic_results_past_successful_items) 

    newly_processed_main_data_buffer: List[Dict[str, Any]] = []
    newly_processed_ic_results_buffer: List[Dict[str, Any]] = []

    total_in_this_session = len(items_to_process_this_run)
    success_count_this_session = 0
    error_count_this_session = 0

    try:
        with tqdm(total=total_in_this_session, desc="处理图片中") as pbar:
            for i in range(0, total_in_this_session, args.concurrency):
                batch_input_items = items_to_process_this_run[i : i + args.concurrency]
                
                api_tasks = []
                # items_in_current_batch_for_api = [] # 这个列表在您的代码中没有被使用，暂时移除

                for item_data_loop_var in batch_input_items: # 修改变量名以避免与外部作用域的 item_data 混淆
                    filename, prompt_text, potential_ics = extract_prompt_info_for_item(item_data_loop_var, hico_object_to_actions)
                    
                    if not prompt_text: 
                        print(f"警告: 文件 {filename} 因为 prompt_text 为空 (可能没有对象或有效描述) 而被跳过，将不会被处理或保存。")
                        continue 

                    img_path = os.path.join(args.image_base_dir, filename)
                    
                    # --- 修改开始 ---
                    api_tasks.append(
                        asyncio.create_task(
                            _get_and_package_api_result( # 调用新的包装函数
                                client, system_task_prompt,
                                img_path,
                                prompt_text, # 这是 user_prompt_text_for_item
                                args.model, args.max_tokens,
                                # 将需要打包的数据传递给包装函数
                                filename,
                                potential_ics,
                                item_data_loop_var # 传递当前循环的 item_data
                            )
                        )
                    )
                    # --- 修改结束 ---
                    # items_in_current_batch_for_api.append(item_data_loop_var) # 如果需要，可以恢复

                if not api_tasks: 
                    pass 

                if api_tasks: 
                    batch_api_call_results_tuples = await asyncio.gather(*api_tasks)
                else:
                    batch_api_call_results_tuples = [] 

                for res_filename, gpt_content, detailed_analysis, api_call_state, res_potential_ics, original_item_from_res in batch_api_call_results_tuples:
                    ic_result_entry = {
                        "filename": res_filename,
                        "ic_gpt_output": gpt_content if api_call_state == "success" else None, 
                        "detailed_analysis": detailed_analysis,
                        "potential_original_ics": res_potential_ics, 
                        "state": api_call_state 
                    }
                    newly_processed_ic_results_buffer.append(ic_result_entry)

                # TODO 待处理
                    if api_call_state == "success" and gpt_content is not None:
                        updated_main_item = update_item_with_gpt_result(original_item_from_res, gpt_content, res_potential_ics)
                    else: 
                        updated_main_item = deepcopy(original_item_from_res)
                        updated_main_item["state"] = api_call_state 
                    
                    # 新增
                    updated_main_item["detailed_analysis"] = detailed_analysis
                    
                    newly_processed_main_data_buffer.append(updated_main_item)

                    if updated_main_item["state"] == "success":
                        success_count_this_session += 1
                    else:
                        error_count_this_session += 1
                
                pbar.update(len(batch_input_items)) # 无论内部是否跳过，批次大小是固定的

                is_last_batch = (i + args.concurrency >= total_in_this_session)
                if (len(newly_processed_main_data_buffer) >= args.save_interval) or \
                   (is_last_batch and newly_processed_main_data_buffer): 
                    
                    current_main_output_data.extend(newly_processed_main_data_buffer)
                    current_ic_results_data.extend(newly_processed_ic_results_buffer)
                    
                    print(f"\n保存点: 新处理 {len(newly_processed_main_data_buffer)} 个项目。")
                    try:
                        save_results_atomically(current_main_output_data, args.output_main_file)
                        print(f"  主输出文件已保存 ({len(current_main_output_data)} 条)。")
                        save_results_atomically(current_ic_results_data, args.output_ic_result_file)
                        print(f"  IC 结果文件已保存 ({len(current_ic_results_data)} 条)。")
                        
                        newly_processed_main_data_buffer.clear()
                        newly_processed_ic_results_buffer.clear()
                    except Exception as e:
                        print(f"  保存点保存失败: {e}")

    finally:
        await client.close()
        print("OpenAI 客户端已关闭。")

    if newly_processed_main_data_buffer: 
        current_main_output_data.extend(newly_processed_main_data_buffer)
        current_ic_results_data.extend(newly_processed_ic_results_buffer)
        newly_processed_main_data_buffer.clear() 
        newly_processed_ic_results_buffer.clear()

    print("最终保存结果...")
    try:
        save_results_atomically(current_main_output_data, args.output_main_file)
        save_results_atomically(current_ic_results_data, args.output_ic_result_file)
        print("所有结果已最终保存。")
    except Exception as e:
        print(f"最终保存失败: {e}")

    print("\n--- 处理总结 ---")
    print(f"本轮处理项目数 (尝试进行API调用或因prompt为空跳过的项目): {total_in_this_session}") # total_in_this_session 是最初计划处理的
    actual_api_attempts = success_count_this_session + error_count_this_session # 实际进行API调用（或编码失败）的
    print(f"  实际API调用/处理尝试数: {actual_api_attempts}")
    print(f"    其中成功: {success_count_this_session}")
    print(f"    其中失败: {error_count_this_session}")
    skipped_due_to_empty_prompt = total_in_this_session - actual_api_attempts
    if skipped_due_to_empty_prompt > 0:
        print(f"  因prompt_text为空而跳过的项目数: {skipped_due_to_empty_prompt}")
    print(f"主输出文件 '{args.output_main_file}' 总条目数: {len(current_main_output_data)}")
    print(f"IC结果文件 '{args.output_ic_result_file}' 总条目数: {len(current_ic_results_data)}")


def main_cli():
    args = Action_Reassignment_arguments() # 使用您定义的参数解析函数
    
    if not args.api_key: # 确保 p3_arguments() 返回的对象有 api_key 属性，或在此之前检查
        print("错误：必须提供 OpenAI API 密钥。")
        return

    for file_path in [args.output_main_file, args.output_ic_result_file]:
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir): 
            os.makedirs(output_dir, exist_ok=True)
            print(f"已创建目录: {output_dir}")

    print("参数配置:")
    for arg_name, value in vars(args).items(): # 使用 arg_name 避免与外部 arg 冲突
        if arg_name == 'api_key' and value is not None: 
             print(f"  {arg_name}: {'*' * 10 if value else '未设置'}")
        else:
            print(f"  {arg_name}: {value}")
    print("-" * 30)

    try: # 添加 try-except 包裹 asyncio.run
        asyncio.run(process_all_data_async(args))
    except Exception as e:
        print(f"脚本主函数执行过程中发生未处理的致命错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_cli()