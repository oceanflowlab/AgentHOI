import json
from pipe_argu import combine_same_HOI
#  将OC、HD、OD一样的进行重组

def merge_identical_interactions(data):
    """
    合并具有相同Object Category、Human Description和Object Description的项的Interaction Class

    Args:
        data (dict): 包含场景信息的字典

    Returns:
        dict: 处理后的字典，相同项的Interaction Class被合并
    """
    result = data.copy()

    # 创建一个字典来存储唯一项
    unique_items = {}

    # 用于存储需要保留的索引
    indices_to_keep = set()

    # 遍历所有项
    for i in range(len(data["Object Category"])):
        # 创建一个键来标识唯一项
        key = (
            data["Object Category"][i],
            data["Human Description"][i],
            data["Object Description"][i]
        )

        if key in unique_items:
            print(data['filename'])
            # 如果已经存在这个项，合并Interaction Class
            existing_idx = unique_items[key]
            current_interactions = set(data["Interaction Class"][i])
            existing_interactions = set(result["Interaction Class"][existing_idx])
            # 合并并去重
            result["Interaction Class"][existing_idx] = list(existing_interactions | current_interactions)
        else:
            # 如果是新项，添加到唯一项字典中
            unique_items[key] = i
            indices_to_keep.add(i)

    # 只保留唯一项
    indices_to_keep = sorted(list(indices_to_keep))
    result["Object Category"] = [data["Object Category"][i] for i in indices_to_keep]
    result["Human Description"] = [data["Human Description"][i] for i in indices_to_keep]
    result["Object Description"] = [data["Object Description"][i] for i in indices_to_keep]
    result["Interaction Class"] = [result["Interaction Class"][i] for i in indices_to_keep]
    
    
    return result


args = combine_same_HOI()

with open(args.input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

newdata = []
for i, item in enumerate(data):
    if len(item['Object Category']) > 0:
        data[i] = merge_identical_interactions(item)
        newdata.append(data[i])

with open(args.output_file, "w", encoding="utf-8") as f:
    json.dump(newdata, f, ensure_ascii=False, indent=2)