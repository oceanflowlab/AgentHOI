import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from datasets import build_evaluator
from datasets.hico_categories import *
import json

from utils.PostProcess import PostProcess
from arguments import get_args_parser

import pandas as pd

import torch

def prepare_dataset_text():
    texts = []
    text_mapper = {}
    for i, hoi in enumerate(HICO_INTERACTIONS):
        action_name = " ".join(hoi["action"].split("_"))
        object_name = hoi["object"]
        s = [action_name, object_name]
        text_mapper[len(texts)] = i
        texts.append(s)
    return texts, text_mapper



def extract_img_id(filename):
    """
    从HICO数据集的文件名中提取img_id

    参数:
        filename: str, 形如 "HICO_test2015_00000033.jpg" 的文件名

    返回:
        int: 提取出的img_id数字
    """
    # 用split先按'_'分割,取最后一个部分 "00000033.jpg"
    id_part = filename.split('_')[-1]

    # 去掉.jpg后缀
    id_part = id_part.split('.')[0]

    # 将字符串转为整数,这会自动去掉前导的0
    img_id = int(id_part)

    # 确保img_id在有效范围内
    assert 1 <= img_id <= 9767, f"Invalid img_id: {img_id}, should be between 1 and 9768"

    return img_id

def evaluate(input_file, postprocessors, args):
    evaluator = build_evaluator(args)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    _, text_mapper = prepare_dataset_text()

    hoi_mapper = {(x["action"], x["object"]): x["interaction_id"] for x in HICO_INTERACTIONS}

    object_to_actions = defaultdict(list)
    for interaction in HICO_INTERACTIONS:
        object_to_actions[interaction['object']].append(interaction['action'])

    for item in tqdm(data, desc="Processing images"):
        if len(item['Box']) == 0:
            continue
        n = len(item["Object Category"])
        
        # print(f"Processing {item['filename']} with {n} boxes")
        img_id = extract_img_id(item['filename'])
        for idx in range(n):
            humanBox = np.array(item["Box"][idx]["humanBox"])
            objectBox = np.array(item["Box"][idx]["objBox"])
            # print(objectBox)
            # 检查是否为空数组
            if humanBox.size == 0 or objectBox.size == 0:
                continue
            

            logits_per_hoi = np.zeros([600, 1])
            m = len(item["Interaction Class"][idx])
            actions = object_to_actions[item["Object Category"][idx]]
            potential_hoitext = [hoi_mapper.get((act, item["Object Category"][idx]))for act in actions]

            for idx2 in range(m):
                hoi_text = (item["Interaction Class"][idx][idx2], item["Object Category"][idx])
                
                ic_logit = item["IC Logits"][idx][idx2]
                obj_logit = item["Object Logits"][idx]
                
                hoi_id = hoi_mapper.get(hoi_text, -1)

                if hoi_id != -1:
                    logits_per_hoi[hoi_id] = (ic_logit + obj_logit) / 2

            for idx3 in range(len(potential_hoitext)):
                hoi_id = potential_hoitext[idx3]
                if hoi_id != -1 and logits_per_hoi[hoi_id] < (ic_logit + obj_logit) / 2.5:
                    logits_per_hoi[hoi_id] = (ic_logit + obj_logit) / 4



            humanBox_scores = humanBox[-1]
            objectBox_scores = objectBox[-1]
            box_scores = humanBox_scores * objectBox_scores

            # 删除最后一列 也就是置信度
            humanBox_without_last_col = humanBox[0:-1]
            objectBox_without_last_col = objectBox[0:-1]
            # 拼接在一起
            pred_boxes = np.concatenate((humanBox_without_last_col, objectBox_without_last_col), axis=-1)

            logits_per_hoi = torch.tensor(logits_per_hoi).to(args.device).type(torch.float).reshape(-1, 600)
            pred_boxes = torch.tensor(pred_boxes).to(args.device).reshape(-1, 8)
            box_scores = torch.tensor(box_scores).to(args.device).reshape(-1, 1)
            # print("img_id:", img_id)
            result = {int(img_id): postprocessors(
                {'pred_logits': logits_per_hoi, 'pred_boxes': pred_boxes, 'box_scores': box_scores},
                text_mapper)}

            evaluator.update(result)

    evaluator.save_preds()
    evaluator.accumulate()
    evaluator.summarize()

    return evaluator


# 使用示例
if __name__ == "__main__":

    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    parser.add_argument(
        "--input_file",
        default="hico_pipe/output/4o/4o_box.json",
        help="Path to AgentHOI prediction JSON with GroundingDINO boxes.",
    )

    args = parser.parse_args()
 
    postprocessors = PostProcess(args.test_score_thresh, args.bbox_lambda, args.enable_softmax)

    evaluate(args.input_file, postprocessors, args)







