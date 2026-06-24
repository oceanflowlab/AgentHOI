import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from datasets import build_evaluator
from datasets.swig import generate_text, key_idxs, prepare_dataset_text
from datasets.swig_v1_categories import SWIG_CATEGORIES, SWIG_ACTIONS, SWIG_INTERACTIONS
import json

from utils.PostProcess_optimized import PostProcess
from arguments import get_args_parser


def evaluate_final_optimized(input_file, postprocessors, args):
    evaluator = build_evaluator(args)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    _, text_mapper = prepare_dataset_text('val')
    reversed_mapper = {value: key for key, value in text_mapper.items()}

    id2obj = {obj["id"]: obj["name"] for obj in SWIG_CATEGORIES}
    id2act = {act["id"]: act["name"] for act in SWIG_ACTIONS}

    hoi_mapper = {(id2act[interaction["action_id"]], id2obj[interaction["object_id"]]): interaction["id"]
                  for interaction in SWIG_INTERACTIONS if interaction['evaluation'] == 1}
    
    # 创建对象到可能动作的映射
    object_to_actions = defaultdict(list)
    for interaction in SWIG_INTERACTIONS:
        if interaction['evaluation'] == 1:
            obj = id2obj[interaction["object_id"]]
            act = id2act[interaction["action_id"]]
            object_to_actions[obj].append(act)

    # 收集所有结果用于批量更新evaluator
    all_results = {}
    
    print("处理数据...")
    for item in tqdm(data, desc="Processing images"):
        if len(item['Box']) == 0:
            continue
            
        n = len(item["Object Category"])
        img_id = int(item["img_id"])
        
        if img_id not in all_results:
            all_results[img_id] = []
        
        # 为当前图片收集所有boxes的数据
        img_logits_list = []
        img_boxes_list = []
        img_box_scores_list = []
        
        for idx in range(n):
            logits_per_hoi = np.zeros(5539, dtype=np.float32)

            try:
                act = item["Interaction Class"][idx]
            except IndexError:
                act = None

            obj = item["Object Category"][idx]
            
            obj_logit = item["Object Logits"][idx]
            ic_logit = item["IC Logits"][idx]
            
            actions = object_to_actions[obj]
            potential_hoitext = [reversed_mapper.get(hoi_mapper.get((action, obj)), -1) for action in actions]
            hoi_text = (act, obj)
            hoi_id = hoi_mapper.get(hoi_text, -1)
            if hoi_id != -1:
                hoi_id_mapped = reversed_mapper.get(hoi_id, -1)
                if hoi_id_mapped != -1:
                    logits_per_hoi[hoi_id_mapped] = (ic_logit + obj_logit) / 2

            # 处理potential HOI
            potential_hoitext = np.array(potential_hoitext)
            valid_indices = potential_hoitext != -1
            valid_hoi_ids = potential_hoitext[valid_indices]
            update_indices = valid_hoi_ids[logits_per_hoi[valid_hoi_ids] < (ic_logit + obj_logit) / 2.5]
            logits_per_hoi[update_indices] = (ic_logit + obj_logit) / 5

            humanBox = np.array(item["Box"][idx]["humanBox"])
            objectBox = np.array(item["Box"][idx]["objBox"])

            humanBox_scores = humanBox[-1]
            objectBox_scores = objectBox[-1]
            box_scores = humanBox_scores * objectBox_scores

            humanBox_without_last_col = humanBox[:-1]
            objectBox_without_last_col = objectBox[:-1]
            pred_boxes = np.concatenate((humanBox_without_last_col, objectBox_without_last_col), axis=-1)

            img_logits_list.append(logits_per_hoi)
            img_boxes_list.append(pred_boxes)
            img_box_scores_list.append(box_scores)
        
        # 批量处理当前图片的所有boxes - 优化版本
        if img_logits_list:
            # 直接处理，不使用PostProcess的复杂逻辑
            for j in range(len(img_logits_list)):
                logits_per_hoi = img_logits_list[j]
                pred_boxes = img_boxes_list[j]
                box_scores = img_box_scores_list[j]
                
                # 找到分数超过阈值的HOI
                valid_hoi_indices = np.where(logits_per_hoi > 0)[0]
                
                for hoi_idx in valid_hoi_indices:
                    if hoi_idx in text_mapper:
                        # 计算最终分数 (简化PostProcess逻辑)
                        hoi_score = logits_per_hoi[hoi_idx]
                        final_score = hoi_score * (box_scores ** args.bbox_lambda)
                        
                        if final_score > args.test_score_thresh:
                            # 直接构造结果
                            mapped_hoi_id = text_mapper[hoi_idx]
                            result = [mapped_hoi_id, float(final_score)] + pred_boxes.tolist()
                            all_results[img_id].append(result)

    # 批量更新evaluator
    print("更新评估器...")
    batch_size = 100  # 每次更新100张图片的结果
    img_ids = list(all_results.keys())
    
    for i in tqdm(range(0, len(img_ids), batch_size), desc="Updating evaluator"):
        batch_end = min(i + batch_size, len(img_ids))
        batch_results = {}
        
        for j in range(i, batch_end):
            img_id = img_ids[j]
            batch_results[img_id] = all_results[img_id]
            
        evaluator.update(batch_results)

    evaluator.save_preds()
    evaluator.accumulate()
    evaluator.summarize()

    return evaluator


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    parser.add_argument(
        "--input_file",
        default="swig_pipe/output/4o/4o_box.json",
        help="Path to AgentHOI prediction JSON with GroundingDINO boxes.",
    )
    args = parser.parse_args()

    postprocessors = PostProcess(args.test_score_thresh, args.bbox_lambda, args.enable_softmax)

    print("使用最终优化版本评估...")
    evaluate_final_optimized(args.input_file, postprocessors, args)
