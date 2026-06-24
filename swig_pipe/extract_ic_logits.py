import json
import os
from typing import List, Dict, Any, Tuple, Optional
from pipe_argu import extract_ic_logits_arguments
import argparse

def extract_interaction_logits(detailed_analysis: List[Dict], interaction_classes: List[List[str]]) -> List[List[float]]:
    """
    从 detailed_analysis 中提取每个 Interaction Class 对应的 logits
    
    Args:
        detailed_analysis: 包含 token 和 probability 的列表
        interaction_classes: 每个 HOP 的 IC 列表
    
    Returns:
        每个 HOP 的 IC logits 列表
    """
    if not detailed_analysis or not interaction_classes:
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
    
    # 处理每个 HOP 的 IC 列表
    all_ic_logits = []
    global_used_token_ranges = set()  # 全局已使用的 token 范围
    
    for hop_idx, hop_ics in enumerate(interaction_classes):
        hop_ic_logits = []
        hop_used_ranges = set()  # 当前 HOP 内已使用的范围
        
        for ic_idx, ic in enumerate(hop_ics):
            ic_logits = find_ic_logits_in_sequence_ordered(
                ic, tokens, probabilities, token_sequence, 
                global_used_token_ranges, hop_used_ranges, 
                hop_idx, ic_idx
            )
            hop_ic_logits.append(ic_logits)
        
        # 将当前 HOP 使用的范围加入全局范围
        global_used_token_ranges.update(hop_used_ranges)
        all_ic_logits.append(hop_ic_logits)
    
    return all_ic_logits

def find_ic_logits_in_sequence_ordered(target_ic: str, tokens: List[str], probabilities: List[float],
                                     token_sequence: str, global_used_ranges: set, 
                                     hop_used_ranges: set, hop_idx: int, ic_idx: int) -> float:
    """
    在 token 序列中找到目标 IC 对应的 logits（按顺序处理）
    
    Args:
        target_ic: 目标交互类别名称
        tokens: token 列表
        probabilities: 对应的概率列表
        token_sequence: 连接后的 token 序列
        global_used_ranges: 全局已使用的 token 范围
        hop_used_ranges: 当前 HOP 内已使用的范围
        hop_idx: 当前 HOP 索引
        ic_idx: 当前 IC 在 HOP 内的索引
    
    Returns:
        该 IC 的 logits（多个 token 时取最小值）
    """
    if not target_ic:
        return 0.0
    
    # 清理目标 IC 名称
    target_cleaned = target_ic.strip()
    
    # 生成可能的匹配候选
    candidates = generate_ic_candidates(target_cleaned)
    
    # 在 token 序列中搜索匹配
    for candidate in candidates:
        match_positions = find_all_matches(token_sequence, candidate)
        
        # 按位置顺序处理匹配
        for start_pos in sorted(match_positions):
            end_pos = start_pos + len(candidate) - 1
            
            # 转换为 token 索引
            start_token_idx, end_token_idx = convert_position_to_token_indices(
                start_pos, end_pos, tokens
            )
            
            if start_token_idx is not None and end_token_idx is not None:
                # 检查是否与全局已使用的范围重叠
                if is_range_overlapping(start_token_idx, end_token_idx, global_used_ranges):
                    continue
                
                # 检查是否与当前 HOP 内已使用的范围重叠（只在同一个 HOP 内避免重复）
                if is_range_overlapping(start_token_idx, end_token_idx, hop_used_ranges):
                    continue
                
                # 获取这个范围内的所有 probabilities，取最小值
                range_probabilities = probabilities[start_token_idx:end_token_idx + 1]
                if range_probabilities:
                    # 标记这个范围为当前 HOP 已使用
                    hop_used_ranges.add((start_token_idx, end_token_idx))
                    return min(range_probabilities)
    
    # 如果都没找到，尝试更宽松的匹配策略
    # 对于像 "hold" 这样的重复词，允许在不同 HOP 间重复使用
    if hop_idx > 0:  # 如果不是第一个 HOP，尝试忽略全局限制
        for candidate in candidates:
            match_positions = find_all_matches(token_sequence, candidate)
            
            for start_pos in sorted(match_positions):
                end_pos = start_pos + len(candidate) - 1
                
                start_token_idx, end_token_idx = convert_position_to_token_indices(
                    start_pos, end_pos, tokens
                )
                
                if start_token_idx is not None and end_token_idx is not None:
                    # 只检查是否与当前 HOP 内已使用的范围重叠
                    if is_range_overlapping(start_token_idx, end_token_idx, hop_used_ranges):
                        continue
                    
                    range_probabilities = probabilities[start_token_idx:end_token_idx + 1]
                    if range_probabilities:
                        hop_used_ranges.add((start_token_idx, end_token_idx))
                        return min(range_probabilities)
    
    # 如果都没找到，返回一个默认值
    print(f"Warning: Could not find logits for IC '{target_ic}' in HOP {hop_idx}, IC index {ic_idx}")
    return 0.0

def generate_ic_candidates(ic_name: str) -> List[str]:
    """
    生成 IC 的可能匹配候选
    
    Args:
        ic_name: 原始 IC 名称
    
    Returns:
        可能的匹配候选列表，按优先级排序
    """
    candidates = []
    
    # 原始名称（最高优先级）
    candidates.append(ic_name)
    
    # 处理下划线情况
    if '_' in ic_name:
        candidates.append(ic_name.replace('_', ' '))
        # 对于 sit_on 这样的情况，可能在 token 中显示为 "sit" + "_on"
        parts = ic_name.split('_')
        if len(parts) == 2:
            candidates.append(parts[0] + '_' + parts[1])
            candidates.append(parts[0] + ' ' + parts[1])
            # 分别匹配两部分
            candidates.append(parts[0])
            candidates.append(parts[1])
    
    # 处理空格情况
    if ' ' in ic_name:
        candidates.append(ic_name.replace(' ', '_'))
        parts = ic_name.split(' ')
        if len(parts) == 2:
            candidates.append(parts[0])
            candidates.append(parts[1])
    
    # 去重并保持顺序
    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered_candidates.append(candidate)
    
    return ordered_candidates

def find_all_matches(text: str, pattern: str) -> List[int]:
    """
    在文本中找到所有匹配模式的起始位置
    
    Args:
        text: 搜索文本
        pattern: 匹配模式
    
    Returns:
        所有匹配的起始位置列表
    """
    matches = []
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        matches.append(pos)
        start = pos + 1
    return matches

def is_range_overlapping(start: int, end: int, used_ranges: set) -> bool:
    """
    检查给定范围是否与已使用的范围重叠
    
    Args:
        start: 起始 token 索引
        end: 结束 token 索引
        used_ranges: 已使用的范围集合
    
    Returns:
        是否重叠
    """
    for used_start, used_end in used_ranges:
        if not (end < used_start or start > used_end):
            return True
    return False

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
                interaction_classes = item.get("Interaction Class", [])
                detailed_analysis = item.get("detailed_analysis", [])
                state = item.get("state")

                if filename is None or state is None:
                    missing_keys = [k for k in ["filename", "state"] if item.get(k) is None]
                    print(f"项目缺少键: {missing_keys} - 项目内容: {str(item)[:200]}...")
                    if filename:
                        failed_files.append(str(filename))
                    continue

                filename = str(filename)

                if state != "success":
                    failed_files.append(filename)
                    continue

                # 提取 Interaction Class 对应的 logits
                interaction_logits = extract_interaction_logits(detailed_analysis, interaction_classes)

                # 保留原有数据并添加新字段
                result_item = dict(item)  # 复制原有数据
                result_item["IC Logits"] = interaction_logits  # 添加新字段
                # 丢弃原有的 detailed_analysis 字段
                result_item.pop("detailed_analysis", None)
                
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
    args = extract_ic_logits_arguments()
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