# 第2步

import json
import re

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from collections import defaultdict


import os
import sys
from tqdm import tqdm
from pipe_argu import filter_object_arguments
import argparse

from swig_v1_categories import SWIG_CATEGORIES


class SWIGMapper:
    def __init__(self):

    
        # 创建对象名称列表和对象到ID的映射
        self.object_names = [obj["name"] for obj in SWIG_CATEGORIES]

    def _normalize_category(self, word: str) -> str:
        """
        标准化交互物体种类字符串
        例如:
        """
        # 将_替换成空格
        if word is None:
            return word

        return word.replace("_", " ").lower()

    def map_object_category(self, object_category: str) -> Tuple[str, float]:
        """将输入的对象类别映射到HICO对象"""

        if object_category in self.object_names:
            return object_category, 1

        normalize = self._normalize_category(object_category)
        if normalize in self.object_names:
            return normalize, 1

        print(object_category)
        return  normalize, 0


def process_results(input_file: str) -> List[Dict]:
    """处理结果文件并进行映射"""
    mapper = SWIGMapper()

    with open('data/swig/swig_test_1000.json', 'r', encoding='utf-8') as f:
        gt = json.load(f)

    gt_file2imgId = {item['file_name']: item['img_id'] for item in gt}
    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    processed_results = []

    for result in tqdm(results):
        
        processed_item = {
            "filename": result["filename"],
            "Object Category": [],
            "Object Logits": [],
            "Interaction Class": [],
            "Human Description": [],
            "Object Description":[],
            "img_id": gt_file2imgId[result["filename"]]
        }

        # 映射每个对象类别和交互
        for obj_cat, obj_log, inter_classes, HD, OD in zip(result["Object Category"], result["Object Logits"], result["Interaction Class"], result["Human Description"], result["Object Description"]):
            # 映射对象类别
            mapped_obj, flag = mapper.map_object_category(obj_cat)

            if flag == 0:
                print(result["filename"])
                continue

            processed_item["Object Category"].append(mapped_obj)
            processed_item["Object Logits"].append(obj_log)
            processed_item["Interaction Class"].append(inter_classes)
            processed_item["Human Description"].append(HD)
            processed_item["Object Description"].append(OD)

        processed_results.append(processed_item)
    return processed_results


def save_processed_results(processed_results: List[Dict], output_file: str):
    """保存处理后的结果"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)


def main():

    args = filter_object_arguments()
    
    input_file = args.input_file
    output_file = args.output_file
    # 处理结果
    processed_results = process_results(input_file)

    # 保存结果
    save_processed_results(processed_results, output_file)

    print(f"处理完成！结果已保存到 {output_file}")


if __name__ == "__main__":
    main()