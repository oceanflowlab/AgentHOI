import json
import os
from typing import List, Dict, Any, Tuple, Optional
from pipe_argu import structured_answer_arguments
import argparse

def extract_object_logits(detailed_analysis: List[Dict], analysis: str) -> List[float]:
    """
    从 detailed_analysis 中提取每个 Object Category 对应的 logits
    
    Args:
        detailed_analysis: 包含 token 和 probability 的列表
        analysis: 原始分析字符串，用于确定 Object Category 的数量和顺序
    
    Returns:
        每个 Object Category 对应的 logits 列表
    """
    if not detailed_analysis or not analysis:
        return []
    
    # 解析 analysis 获取 Object Category 列表
    parsed_data = parse_analysis_string(analysis)
    object_categories = parsed_data["Object Category"]
    
    if not object_categories:
        return []
    
    # 构建 token 序列和对应的 probability
    tokens = []
    probabilities = []
    
    for item in detailed_analysis:
        if 'token' in item and 'probability' in item:
            tokens.append(item['token'])
            probabilities.append(item['probability'])
    
    if not tokens:
        return []
    
    # 将 tokens 连接成字符串，用于匹配
    token_sequence = ''.join(tokens)
    
    # 找到所有 "OC:" 的位置，确定 OC 字段的范围
    oc_ranges = find_oc_field_ranges(token_sequence, tokens)
    
    object_logits = []
    used_oc_indices = set()  # 记录已经使用过的 OC 索引，避免重复匹配
    
    # 为每个 Object Category 按顺序查找对应的 logits
    for obj_category in object_categories:
        obj_logits = find_object_logits_in_oc_fields(
            obj_category, tokens, probabilities, token_sequence, oc_ranges, used_oc_indices
        )
        object_logits.append(obj_logits)
    
    return object_logits

def find_oc_field_ranges(token_sequence: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """
    找到所有 OC 字段的范围（从 "OC:" 开始到分号 ";" 结束）
    
    Args:
        token_sequence: 连接后的 token 序列
        tokens: token 列表
    
    Returns:
        List of (start_pos, end_pos) tuples indicating OC field ranges in token_sequence
    """
    oc_ranges = []
    start_pos = 0
    
    while True:
        # 找到下一个 "OC:" 的位置
        oc_pos = token_sequence.find("OC:", start_pos)
        if oc_pos == -1:
            break
        
        # 找到这个 OC 字段的结束位置（下一个分号）
        semicolon_pos = token_sequence.find(";", oc_pos)
        if semicolon_pos == -1:
            # 如果没有找到分号，可能是最后一个字段，搜索到字符串结尾
            semicolon_pos = len(token_sequence)
        
        # OC 字段的内容是从 "OC:" 之后到分号之前
        oc_content_start = oc_pos + 3  # "OC:" 长度为3
        oc_content_end = semicolon_pos
        
        if oc_content_start < oc_content_end:
            oc_ranges.append((oc_content_start, oc_content_end))
        
        start_pos = semicolon_pos + 1
    
    return oc_ranges

def find_object_logits_in_oc_fields(target_object: str, tokens: List[str], probabilities: List[float], 
                                   token_sequence: str, oc_ranges: List[Tuple[int, int]], used_oc_indices: set) -> float:
    """
    在 OC 字段范围内找到目标 object 对应的 logits
    
    Args:
        target_object: 目标对象名称
        tokens: token 列表
        probabilities: 对应的概率列表
        token_sequence: 连接后的 token 序列
        oc_ranges: OC 字段的范围列表
        used_oc_indices: 已经使用过的 OC 字段索引集合
    
    Returns:
        该对象的 logits（多个 token 时取最小值）
    """
    if not target_object:
        return 0.0
    
    # 清理目标对象名称，处理空格和下划线
    target_cleaned = target_object.strip()
    
    # 尝试多种匹配方式
    candidates = [
        target_cleaned,
        target_cleaned.replace(' ', '_'),
        target_cleaned.replace('_', ' ')
    ]
    
    # 在每个 OC 字段范围内搜索
    for oc_index, (oc_start, oc_end) in enumerate(oc_ranges):
        if oc_index in used_oc_indices:
            continue  # 跳过已经使用过的 OC 字段
        
        # 获取这个 OC 字段的内容
        oc_content = token_sequence[oc_start:oc_end]
        
        # 尝试在这个 OC 字段中找到目标对象
        for candidate in candidates:
            if candidate in oc_content:
                # 找到了匹配，现在需要确定对应的 token 位置
                # 在 OC 字段内的相对位置
                relative_pos = oc_content.find(candidate)
                # 在整个 token 序列中的绝对位置
                absolute_start = oc_start + relative_pos
                absolute_end = absolute_start + len(candidate) - 1
                
                # 转换为 token 索引
                start_token_idx, end_token_idx = convert_position_to_token_indices(
                    absolute_start, absolute_end, tokens
                )
                
                if start_token_idx is not None and end_token_idx is not None:
                    # 获取这个范围内的所有 probabilities，取最小值
                    range_probabilities = probabilities[start_token_idx:end_token_idx + 1]
                    if range_probabilities:
                        used_oc_indices.add(oc_index)  # 标记这个 OC 字段为已使用
                        return min(range_probabilities)
    
    # 如果都没找到，返回一个默认值
    print(f"Warning: Could not find logits for object '{target_object}' in OC fields")
    return 0.0

def convert_position_to_token_indices(start_pos: int, end_pos: int, tokens: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """
    将字符位置转换为 token 索引
    
    Args:
        start_pos: 开始字符位置
        end_pos: 结束字符位置
        tokens: token 列表
    
    Returns:
        (start_token_idx, end_token_idx) 或 (None, None) 如果找不到
    """
    current_pos = 0
    start_token_idx = None
    end_token_idx = None
    
    for i, token in enumerate(tokens):
        token_start = current_pos
        token_end = current_pos + len(token) - 1
        
        # 找到开始的 token
        if start_token_idx is None and token_start <= start_pos <= token_end:
            start_token_idx = i
        
        # 找到结束的 token
        if token_start <= end_pos <= token_end:
            end_token_idx = i
            break
        
        current_pos += len(token)
    
    return start_token_idx, end_token_idx

def parse_analysis_string(analysis: str) -> Dict[str, List[Any]]:
    """
    解析分析字符串，提取所有类别和描述。
    优化后能够处理通过 ';' 或 '\n' 分隔的多个OC记录。
    """
    result = {
        "Object Category": [],
        "Interaction Class": [],
        "Human Description": [],
        "Object Description": []
    }

    if not analysis or not isinstance(analysis, str):
        return result

    # 1. 将换行符统一为分号，并清理字符串
    processed_analysis = analysis.replace('\n', ';').strip()
    while ';;' in processed_analysis:
        processed_analysis = processed_analysis.replace(';;', ';')
    processed_analysis = processed_analysis.strip(';')

    if not processed_analysis:
        return result

    # 2. 以 "OC:" 作为记录的起始标志进行切分
    raw_oc_blocks = processed_analysis.split('OC:')

    for block_text in raw_oc_blocks:
        block_text = block_text.strip()
        if not block_text:
            continue

        parts = [p.strip() for p in block_text.split(';', 3)]

        oc_val = ""
        ic_val_list = []
        hd_val = ""
        od_val = ""

        if not parts or not parts[0] or len(parts) < 4:
            print(parts)
            continue

        oc_val = parts[0]

        # 解析 IC
        if len(parts) > 1 and parts[1].upper().startswith("IC:"):
            ic_str = parts[1][len("IC:"):].strip()
            if ic_str:
                ic_val_list = [val.strip() for val in ic_str.split(' ') if val.strip()]

        # 解析 HD
        if len(parts) > 2 and parts[2].upper().startswith("HD:"):
            hd_val = parts[2][len("HD:"):].strip()

        # 解析 OD
        if len(parts) > 3 and parts[3].upper().startswith("OD:"):
            od_val = parts[3][len("OD:"):].strip().strip(';')

        result["Object Category"].append(oc_val)
        result["Interaction Class"].append(ic_val_list)
        result["Human Description"].append(hd_val)
        result["Object Description"].append(od_val)

    return result

def fix_json_content(content: str) -> str:
    """修复JSON内容的格式问题"""
    content = content.strip()
    content = content.replace('\n', '')
    content = content.replace('\r', '')
    content = content.replace('}{', '},{')

    if not content.startswith('['):
        content = '[' + content
    if not content.endswith(']'):
        content = content.rstrip(',') + ']'

    return content

def process_json_file(input_file: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """处理JSON文件并返回成功和失败的结果"""
    successful_results = []
    failed_files = []

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        fixed_content = fix_json_content(content)
        data = []
        try:
            data = json.loads(fixed_content)
        except json.JSONDecodeError as e:
            print(f"JSON整体解析错误 (使用fixed_content): {str(e)}")
            print("尝试逐行或累积行解析原始JSON内容...")

            lines = content.split('\n')
            current_item_buffer = ""
            for line_idx, line_content in enumerate(lines):
                line_content_stripped = line_content.strip()
                if not line_content_stripped:
                    if current_item_buffer.strip():
                        try:
                            item = json.loads(current_item_buffer)
                            data.append(item)
                        except json.JSONDecodeError:
                            print(f"跳过无效的JSON对象 (缓冲区在空行处): {current_item_buffer[:100]}...")
                        current_item_buffer = ""
                    continue

                current_item_buffer += line_content

                temp_buffer_stripped = current_item_buffer.strip()
                if temp_buffer_stripped.endswith('}') or temp_buffer_stripped.endswith('},') or (line_idx == len(lines) -1 and temp_buffer_stripped):
                    try:
                        item_to_parse = temp_buffer_stripped
                        item = json.loads(item_to_parse)
                        data.append(item)
                        current_item_buffer = ""
                    except json.JSONDecodeError:
                        if line_idx == len(lines) -1:
                            print(f"跳过无效的JSON对象 (最终缓冲区): {current_item_buffer[:100]}...")

            if current_item_buffer.strip():
                try:
                    item = json.loads(current_item_buffer)
                    data.append(item)
                except json.JSONDecodeError:
                    print(f"跳过无效的JSON对象 (末尾残留缓冲区): {current_item_buffer[:100]}...")

        if not isinstance(data, list):
            data = [data]

        for item in data:
            try:
                if not isinstance(item, dict):
                    print(f"跳过非字典类型的项目: {type(item)}")
                    continue

                filename = item.get("filename")
                analysis_str = item.get("analysis")
                detailed_analysis = item.get("detailed_analysis", [])
                state = item.get("state")

                if filename is None or analysis_str is None or state is None:
                    missing_keys = [k for k in ["filename", "analysis", "state"] if item.get(k) is None]
                    print(f"项目缺少键: {missing_keys} - 项目内容: {str(item)[:200]}...")
                    if filename:
                        failed_files.append(str(filename))
                    continue

                filename = str(filename)

                if state != "success":
                    failed_files.append(filename)
                    continue

                parsed_data = parse_analysis_string(analysis_str)
                
                # 提取 Object Category 对应的 logits
                object_logits = extract_object_logits(detailed_analysis, analysis_str)

                result_item = {
                    "filename": filename,
                    "Object Category": parsed_data["Object Category"],
                    "Object Logits": object_logits,  # 新增字段
                    "Interaction Class": parsed_data["Interaction Class"],
                    "Human Description": parsed_data["Human Description"],
                    "Object Description": parsed_data["Object Description"]
                }
                successful_results.append(result_item)

            except Exception as e_item:
                print(f"处理项目时发生内部错误: {str(e_item)} - 项目: {str(item)[:200]}")
                filename_in_item = item.get("filename", "未知文件名")
                failed_files.append(str(filename_in_item))
                continue

    except FileNotFoundError:
        print(f"错误: 输入文件 {input_file} 未找到。")
    except Exception as e_file:
        print(f"处理文件 {input_file} 时发生严重错误: {str(e_file)}")

    return successful_results, failed_files

def save_results(successful_results: List[Dict[str, Any]], failed_files: List[str],
                output_file: str, failed_output_file: str):
    """保存处理结果到文件"""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    failed_dir = os.path.dirname(failed_output_file)
    if failed_dir and not os.path.exists(failed_dir):
        os.makedirs(failed_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(successful_results, f, ensure_ascii=False, indent=2)

    if failed_files:
        with open(failed_output_file, 'w', encoding='utf-8') as f:
            json.dump(failed_files, f, ensure_ascii=False, indent=2)

def main():
    args = structured_answer_arguments()
    successful_results, failed_files_list = process_json_file(args.input_file)
    save_results(successful_results, failed_files_list, args.output_file, args.failed_file)

    print(f"处理完成！成功处理 {len(successful_results)} 个项目。")
    if failed_files_list:
        print(f"失败或跳过 {len(failed_files_list)} 个项目。")
        print(f"失败的项目列表已保存到 {args.failed_file}")
    else:
        print("没有项目处理失败。")

if __name__ == "__main__":
    main()