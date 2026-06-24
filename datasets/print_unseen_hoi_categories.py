# -*- coding: utf-8 -*-
"""打印 hico_unseen_index 中指定划分对应的 HOI 具体类别。"""

from hico_categories import HICO_INTERACTIONS, hico_unseen_index

TARGET_SPLITS = ["rare_first", "non_rare_first", "unseen_object", "unseen_verb"]


def format_hoi(interaction_id: int) -> str:
    hoi = HICO_INTERACTIONS[interaction_id]
    action = hoi["action"]
    obj = hoi["object"]
    return f"{interaction_id:>3}: {action} {obj}"


def main() -> None:
    for split_name in TARGET_SPLITS:
        ids = hico_unseen_index.get(split_name, [])
        print(f"\n===== {split_name} (共 {len(ids)} 个) =====")
        for interaction_id in ids:
            print(format_hoi(interaction_id))


if __name__ == "__main__":
    main()
