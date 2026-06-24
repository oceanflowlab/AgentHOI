import json
from pipe_argu import object_logits_update_arguments
import argparse

# 将第一轮VLM结果中的Object Logits更新到第二轮后的结果

args = object_logits_update_arguments()

with open(args.refer_file, 'r', encoding='utf-8') as f:
    refer_data = json.load(f)

with open(args.input_file, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

# 为 refer_data 创建映射字典，以 filename 为第一层 key
refer_mapping = {}
for item in refer_data:
    filename = item["filename"]
    if filename not in refer_mapping:
        refer_mapping[filename] = {}
    
    # 为每个对象创建组合 key
    for i in range(len(item["Object Category"])):
        combined_key = (
            item["Object Category"][i],
            item["Human Description"][i].strip(),
            item["Object Description"][i].strip()
        )
        refer_mapping[filename][combined_key] = item["Object Logits"][i]

# 更新 input_data 中的 Object Logits
for item in input_data:
    filename = item["filename"]
    
    # 如果该文件在 refer_data 中存在
    if filename in refer_mapping:
        # 检查每个对象
        for i in range(len(item["Object Category"])):
            combined_key = (
                item["Object Category"][i],
                item["Human Description"][i].strip(),
                item["Object Description"][i].strip()
            )
            
            # 如果找到匹配的 key，更新 logits
            if combined_key in refer_mapping[filename]:
                item["Object Logits"][i] = refer_mapping[filename][combined_key]
                print(f"Updated logits for {filename}, object {i}: {item['Object Logits'][i]}")

# 保存更新后的数据
with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(input_data, f, ensure_ascii=False, indent=2)

print(f"Updated data saved to: {args.output_file}")
