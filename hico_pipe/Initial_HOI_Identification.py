import argparse
import asyncio
import base64
import json
import os
import time # 用于备份文件名时间戳
from typing import List, Dict, Set, Optional, Any, Tuple
from openai import AsyncOpenAI
from tqdm import tqdm

from pipe_argu import initial_HOI_Identification_arguments



# --- 配置与设置 ---
def load_task_prompt(prompt_file_path: str) -> str:
    # 从文件加载任务提示
    # 注意：如果文件不存在或无法读取，将直接抛出异常
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        return f.read()

# --- 图片及数据处理函数 ---
def local_image_to_base64(image_path: str) -> Optional[str]:
    # 将本地图片文件转换为Base64编码的字符串
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"图片文件未找到 (用于Base64编码): {image_path}")
        return None
    except Exception as e:
        print(f"读取或编码图片 {image_path} 时出错: {e}")
        return None

def get_all_image_paths_from_annotations(base_image_dir: str, annotation_file_path: str) -> List[str]:
    # 从注解文件 (JSON格式) 读取图片文件名，并构建完整的图片路径
    # 假设注解文件包含一个字典列表，每个字典有 'file_name' 键
    # 注意：如果文件不存在、不是有效JSON或格式不符，将直接抛出异常
    
    print(f"从注解文件 '{annotation_file_path}' 加载图片列表...")
    with open(annotation_file_path, "r", encoding='utf-8') as f:
        annotations = json.load(f)

    image_paths: List[str] = []
    if not isinstance(annotations, list):
        print(f"警告：注解文件 '{annotation_file_path}' 的内容不是预期的列表格式。")
        return image_paths # 或者抛出错误

    for item in annotations:
        if isinstance(item, dict) and 'filename' in item:
            img_path = os.path.join(base_image_dir, item['filename'])
            image_paths.append(img_path)
        else:
            print(f"警告：跳过注解文件中的无效或不完整项: {item}")
    
    print(f"从注解文件找到了 {len(image_paths)} 个图片路径。")
    return image_paths

# --- 结果处理函数 ---
def load_and_filter_results(file_path: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    # 从JSON文件加载结果 (应为一个字典列表)
    # 筛选出状态为 'error' 的项目以进行重处理
    # 返回有效的 (成功状态的) 结果列表及其文件名集合
    
    valid_results: List[Dict[str, Any]] = []
    processed_successful_filenames: Set[str] = set()

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"输出文件 '{file_path}' 未找到或为空。将从头开始。")
        return valid_results, processed_successful_filenames

    print(f"从 '{file_path}' 加载已有的结果...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
    except json.JSONDecodeError:
        print(f"'{file_path}' 中的JSON解码错误。文件可能已损坏。")
        new_name = file_path + ".corrupted_backup_" + time.strftime("%Y%m%d%H%M%S")
        try:
            os.rename(file_path, new_name)
            print(f"已将损坏的文件备份到 '{new_name}'。本次将从头开始。")
        except OSError:
            print(f"无法备份损坏的文件 '{file_path}'。本次将从头开始。")
        return [], set() # 如果文件损坏，则从头开始

    if not isinstance(loaded_data, list):
        print(f"输出文件 '{file_path}' 的内容不是一个JSON列表。将从头开始。")
        return valid_results, processed_successful_filenames

    items_to_reprocess_count = 0
    for item in loaded_data:
        # 假设item结构基本正确，但仍检查关键字段
        filename = item.get("filename", "未知文件")
        current_state = item.get("state")

        if current_state == "success":
            valid_results.append(item)
            processed_successful_filenames.add(filename)
        elif isinstance(current_state, str) and current_state.startswith("error_"):
            # print(f"项目 '{filename}' 的错误状态为 '{current_state}'，将重新处理。") # 此print可选，会比较多信息
            items_to_reprocess_count += 1
        else: # 其他非 "success" 状态也重新处理
            # print(f"项目 '{filename}' 的状态为 '{current_state}'，将重新处理。") # 此print可选
            items_to_reprocess_count += 1
    
    if items_to_reprocess_count > 0:
         print(f"{items_to_reprocess_count} 个项目因错误或其他非成功状态被标记以便重新处理。")

    print(f"从 '{file_path}' 加载了 {len(valid_results)} 个过去的成功结果。")
    return valid_results, processed_successful_filenames

def save_results_atomically(results: List[Dict[str, Any]], target_file_path: str) -> None:
    # 原子性地保存结果列表到目标文件 (通过临时文件和重命名)
    # 注意：如果保存过程中发生IO错误（如磁盘满），将直接抛出异常
    temp_file_path = target_file_path + ".tmp"
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(temp_file_path, target_file_path) # 原子性重命名

# --- OpenAI API 交互 ---

async def analyze_single_image_async(
    client: AsyncOpenAI, 
    task_prompt: str, 
    image_path: str,
    model_name: str,
    max_tokens: int
) -> Dict[str, Any]:
    # 异步分析单个图片
    image_filename = os.path.basename(image_path)
    
    base64_image = local_image_to_base64(image_path)
    if base64_image is None:
        return {
            "filename": image_filename,
            "analysis": f"错误：无法读取或编码图片文件 {image_path}。",
            "state": "error_image_encoding"
        }

    try:
        messages = [
            {"role": "system", "content": task_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "List all HOPs of this image and Your output must be strictly followed by examples and do not output explanations or other content."},
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            ]}
        ]
        
        # --- 主要修改点在这里 ---
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages, # type: ignore
            max_tokens=max_tokens,
            logprobs=True  # <--- 新增参数：请求对数概率
        )
        # --- 修改结束 ---
        
        analysis_content = response.choices[0].message.content
        # import pdb; pdb.set_trace()
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

        return {
            "filename": image_filename,
            "analysis": analysis_content, # 保留原始的完整句子
            "detailed_analysis": detailed_analysis, # 新增带有概率的详细分析
            "state": "success"
        }
    except Exception as e:
        print(f"处理图片 {image_filename} 时API发生错误: {e}")
        return {
            "filename": image_filename,
            "analysis": str(e),
            "detailed_analysis": [],
            "state": "error_api_call"
        }
# --- 主流程控制 ---

async def run_image_analysis(
    image_base_dir: str,
    annotation_file: str,
    prompt_file: str,
    output_file: str,
    concurrency_limit: int,
    save_interval: int,
    api_key: str,
    api_base_url: Optional[str],
    openai_model: str,
    max_tokens_per_image: int
) -> None:
    # 主函数，组织图片分析流程
    
    task_prompt = load_task_prompt(prompt_file) # 如果失败会在此处抛出异常
    
    # 加载已有结果，并筛选出错误项以便重处理
    all_accumulated_results, processed_successful_filenames = load_and_filter_results(output_file)
    
    all_image_full_paths = get_all_image_paths_from_annotations(image_base_dir, annotation_file) # 失败会抛异常
    if not all_image_full_paths:
        print("从注解文件未找到任何图片。正在退出。")
        if not all_accumulated_results and not os.path.exists(output_file):
            save_results_atomically([], output_file) # 如果完全没有结果，确保输出一个空列表的JSON文件
        return

    images_to_process_paths = [
        path for path in all_image_full_paths 
        if os.path.basename(path) not in processed_successful_filenames
    ]

    if not images_to_process_paths:
        print("所有需要处理的图片均已成功处理并记录。")
        # 确保输出文件反映了准确加载和筛选后的结果
        save_results_atomically(all_accumulated_results, output_file)
        return

    print(f"开始分析 {len(images_to_process_paths)} 张新的或需要重新处理的图片。")

    client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)
    
    new_results_current_session_buffer: List[Dict[str, Any]] = []
    successfully_processed_this_run = 0
    failed_this_run = 0

    try:
        with tqdm(total=len(images_to_process_paths), desc="分析图片中") as pbar:
            for i in range(0, len(images_to_process_paths), concurrency_limit):
                batch_image_paths = images_to_process_paths[i : i + concurrency_limit]
                
                async_tasks = [
                    analyze_single_image_async(client, task_prompt, img_path, openai_model, max_tokens_per_image)
                    for img_path in batch_image_paths
                ]
                
                batch_results = await asyncio.gather(*async_tasks)
                
                new_results_current_session_buffer.extend(batch_results)
                all_accumulated_results.extend(batch_results)
                
                pbar.update(len(batch_image_paths))

                is_last_batch_iteration = (i + concurrency_limit >= len(images_to_process_paths))
                
                if (len(new_results_current_session_buffer) >= save_interval) or \
                   (is_last_batch_iteration and new_results_current_session_buffer):
                    print(f"\n保存点：本次已处理 {len(new_results_current_session_buffer)} 个新项目。"
                          f"内存中总项目数: {len(all_accumulated_results)}。正在保存到 '{output_file}'...")
                    save_results_atomically(all_accumulated_results, output_file) # 如果失败会抛异常
                    
                    for res in new_results_current_session_buffer:
                        if res.get("state") == "success":
                            successfully_processed_this_run +=1
                        else:
                            failed_this_run +=1
                    new_results_current_session_buffer.clear() 
    finally:
        await client.close()
        print("OpenAI 客户端已关闭。")

    # 统计最后缓冲区中可能未被计数的项目 (一般在正常结束时缓冲区应为空)
    for res in new_results_current_session_buffer:
        if res.get("state") == "success":
            successfully_processed_this_run +=1
        else:
            failed_this_run +=1
    
    print("正在最终确定结果...")
    save_results_atomically(all_accumulated_results, output_file) # 确保最终状态被保存
    
    print("图片分析流程完成。")
    print(f"输出文件 '{output_file}' 中总项目数: {len(all_accumulated_results)}")
    print(f"本次运行处理情况: {successfully_processed_this_run} 成功, {failed_this_run} 失败。")


def main_cli():
    # 使用从 argument.py 导入的函数来解析参数
    args = initial_HOI_Identification_arguments() # <--- 修改点
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    if not args.api_key:
        print("错误：必须提供OpenAI API密钥。请设置 OPENAI_API_KEY 环境变量或使用 --api_key 参数。")
        return

    print(f"输出将保存到 '{args.output_file}'。保存间隔: 每处理 {args.save_interval} 张新图片。")
    print(f"并发数: {args.concurrency}。模型: '{args.openai_model}'。")

    try:
        asyncio.run(run_image_analysis(
            image_base_dir=args.image_base_dir,
            annotation_file=args.annotation_file,
            prompt_file=args.prompt_file,
            output_file=args.output_file,
            concurrency_limit=args.concurrency,
            save_interval=args.save_interval,
            api_key=args.api_key,
            api_base_url=args.api_base_url,
            openai_model=args.openai_model,
            max_tokens_per_image=args.max_tokens_per_image
        ))
    except Exception as e:
        print(f"脚本执行过程中发生未处理的致命错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_cli()