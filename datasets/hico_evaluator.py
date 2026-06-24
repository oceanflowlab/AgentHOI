import collections
import numpy as np
import json
import os
import pickle
from .hico_categories import HICO_INTERACTIONS, HICO_ACTIONS, HICO_OBJECTS
from .hico_categories import ZERO_SHOT_INTERACTION_IDS, NON_INTERACTION_IDS, hico_unseen_index, RARE_INTERACTION_IDS


class HICOEvaluator(object):
    ''' Evaluator for HICO-DET datasets '''
    def __init__(self, anno_file, output_dir, zero_shot_type, ignore_non_interaction):
        size = 600
        self.size = size
        self.no_interaction_missing_stats = None
        self.gts = self.load_anno(anno_file)
        self.scores = {i: [] for i in range(size)}
        self.boxes = {i: [] for i in range(size)}
        self.keys = {i: [] for i in range(size)}
        self.hico_ap = np.zeros(size)
        self.hico_rec = np.zeros(size)
        self.output_dir = output_dir
        self.zero_shot_type = zero_shot_type
        self.zero_shot_interaction_ids = hico_unseen_index[zero_shot_type]
        self.ignore_non_interaction = ignore_non_interaction
        self.interaction_id2meta = {x["interaction_id"]: x for x in HICO_INTERACTIONS}

    def _count_gt_instances(self, hoi_id):
        if hoi_id not in self.gts:
            return 0
        return int(sum(v.shape[0] for v in self.gts[hoi_id].values()))

    def _split_score_stats(self, hoi_ids):
        scores = []
        for hoi_id in hoi_ids:
            if len(self.scores[hoi_id]) > 0:
                scores.extend(self.scores[hoi_id])

        if len(scores) == 0:
            return {"mean": 0.0, "p50": 0.0, "p90": 0.0}

        arr = np.array(scores, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    def _mean_ap_safe(self, hoi_ids):
        hoi_ids = np.array(hoi_ids, dtype=np.int64)
        if len(hoi_ids) == 0:
            return 0.0
        return float(np.mean(self.hico_ap[hoi_ids]))

    def _print_unseen_map_with_without_no_interaction(self, split_name, hoi_ids):
        hoi_ids = np.array(hoi_ids, dtype=np.int64)
        no_interaction_hois = np.intersect1d(hoi_ids, NON_INTERACTION_IDS)
        without_no_interaction_hois = np.setdiff1d(hoi_ids, NON_INTERACTION_IDS)

        map_with_no_interaction = self._mean_ap_safe(hoi_ids)
        map_without_no_interaction = self._mean_ap_safe(without_no_interaction_hois)
        map_only_no_interaction = self._mean_ap_safe(no_interaction_hois)

        print(
            f"{split_name} mAP w/ no_interaction: {map_with_no_interaction * 100.:.2f} "
            f"(#classes={len(hoi_ids)}, #no_interaction={len(no_interaction_hois)})"
        )
        print(
            f"{split_name} mAP w/o no_interaction: {map_without_no_interaction * 100.:.2f} "
            f"(#classes={len(without_no_interaction_hois)})"
        )
        print(
            f"{split_name} no_interaction-only mAP: {map_only_no_interaction * 100.:.2f} "
            f"(#classes={len(no_interaction_hois)})"
        )

    def _print_no_interaction_missing_stats(self):
        stats = self.no_interaction_missing_stats
        if stats is None:
            print("\n========== no_interaction missing-label statistics ==========")
            print("No no_interaction missing-label statistics are available.")
            return

        print("\n========== no_interaction missing-label statistics ==========")
        print(
            "Definition: for every image, every person(subject)-object pair without any HOI annotation "
            "is counted as a missing no_interaction label."
        )
        print(f"#images: {stats['num_images']}")
        print(f"total HOI annotations: {stats['total_hoi_annotations']}")
        print(f"avg HOI annotations / image: {stats['avg_hoi_annotations_per_image']:.4f}")
        print(f"max HOI annotations in one image: {stats['max_hoi_annotations_per_image']}")
        print(f"image ids with max HOI annotations: {stats['image_ids_with_max_hoi_annotations']}")
        print(f"total candidate subject-object pairs: {stats['total_candidate_pairs']}")
        print(f"total annotated subject-object pairs: {stats['total_annotated_pairs']}")
        print(f"total existing no_interaction annotations: {stats['total_existing_no_interaction_annotations']}")
        print(f"total existing no_interaction subject-object pairs: {stats['total_existing_no_interaction_pairs']}")
        print(f"total missing no_interaction pairs: {stats['total_missing_no_interaction_pairs']}")
        print(f"avg missing no_interaction pairs / image: {stats['avg_missing_no_interaction_pairs_per_image']:.4f}")
        print(f"avg candidate subject-object pairs / image: {stats['avg_candidate_pairs_per_image']:.4f}")
        print(f"dataset missing no_interaction ratio: {stats['dataset_missing_no_interaction_ratio'] * 100.:.2f}%")
        print(f"mean per-image missing no_interaction ratio: {stats['mean_per_image_missing_no_interaction_ratio'] * 100.:.2f}%")

    def _print_split_diagnostics(self, split_name, hoi_ids):
        if len(hoi_ids) == 0:
            print(f"[{split_name}] no classes")
            return

        aps = self.hico_ap[hoi_ids]
        recs = self.hico_rec[hoi_ids]
        gt_counts = np.array([self._count_gt_instances(int(i)) for i in hoi_ids], dtype=np.int64)
        pred_counts = np.array([len(self.scores[int(i)]) for i in hoi_ids], dtype=np.int64)
        score_stats = self._split_score_stats(hoi_ids)

        zero_ap_ratio = float(np.mean(aps == 0))
        zero_rec_ratio = float(np.mean(recs == 0))
        no_pred_ratio = float(np.mean(pred_counts == 0))

        print(f"[{split_name}] #classes={len(hoi_ids)}, #gt={int(gt_counts.sum())}, #pred={int(pred_counts.sum())}")
        print(
            f"[{split_name}] AP(mean/median)={float(np.mean(aps)) * 100:.2f}/{float(np.median(aps)) * 100:.2f}, "
            f"Rec(mean/median)={float(np.mean(recs)) * 100:.2f}/{float(np.median(recs)) * 100:.2f}"
        )
        print(
            f"[{split_name}] zero-AP ratio={zero_ap_ratio * 100:.2f}%, zero-Rec ratio={zero_rec_ratio * 100:.2f}%, "
            f"no-pred ratio={no_pred_ratio * 100:.2f}%"
        )
        print(
            f"[{split_name}] score(mean/p50/p90)="
            f"{score_stats['mean']:.4f}/{score_stats['p50']:.4f}/{score_stats['p90']:.4f}"
        )

    def _print_nf_unseen_bottomk(self, nf_unseen_hois, topk=15):
        rows = []
        for hoi_id in nf_unseen_hois:
            hoi_id = int(hoi_id)
            meta = self.interaction_id2meta.get(hoi_id, {})
            rows.append({
                "hoi_id": hoi_id,
                "action": meta.get("action", "unknown"),
                "object": meta.get("object", "unknown"),
                "ap": float(self.hico_ap[hoi_id]),
                "rec": float(self.hico_rec[hoi_id]),
                "gt": self._count_gt_instances(hoi_id),
                "pred": len(self.scores[hoi_id]),
                "score_mean": float(np.mean(self.scores[hoi_id])) if len(self.scores[hoi_id]) > 0 else 0.0,
            })

        rows = sorted(rows, key=lambda x: (x["ap"], x["rec"], x["gt"]))
        topk = min(topk, len(rows))
        print(f"[NF_unseen] Bottom-{topk} classes by AP:")
        for i in range(topk):
            r = rows[i]
            print(
                f"  {i + 1:02d}. HOI {r['hoi_id']:03d} ({r['action']}, {r['object']}): "
                f"AP={r['ap'] * 100:.2f}, Rec={r['rec'] * 100:.2f}, GT={r['gt']}, Pred={r['pred']}, "
                f"score_mean={r['score_mean']:.4f}"
            )

    def update(self, predictions):
        ''' Store predictions
        Args:
            predictions (dict): a dictionary in the following format.
            {
                img_id: [
                    [hoi_id, score, pbox_x1, pbox_y1, pbox_x2, pbox_y2, obox_x1, obox_y1, obox_x2, obox_y2],
                    ...
                    ...
                ]
            }
        '''
        for img_id, preds in predictions.items():
            for pred in preds:
                hoi_id = pred[0]
                score = pred[1]
                boxes = pred[2:]
                self.scores[hoi_id].append(score)
                self.boxes[hoi_id].append(boxes)
                self.keys[hoi_id].append(img_id)

    def accumulate(self):
        for hoi_id in range(600):
            gts_per_hoi = self.gts[hoi_id]
            ap, rec = calc_ap(self.scores[hoi_id], self.boxes[hoi_id], self.keys[hoi_id], gts_per_hoi)
            self.hico_ap[hoi_id], self.hico_rec[hoi_id] = ap, rec

    def summarize(self):
        if self.ignore_non_interaction:
            valid_hois = np.setdiff1d(np.arange(600), NON_INTERACTION_IDS)
            seen_hois = np.setdiff1d(valid_hois, self.zero_shot_interaction_ids)
            zero_shot_hois = np.setdiff1d(self.zero_shot_interaction_ids, NON_INTERACTION_IDS)

            rare_hois = np.setdiff1d(RARE_INTERACTION_IDS, [])
            non_rare_hois = np.setdiff1d(valid_hois, RARE_INTERACTION_IDS)

            RF_seen_hois = np.setdiff1d(valid_hois, hico_unseen_index["rare_first"])
            RF_unseen_hois = np.setdiff1d(hico_unseen_index["rare_first"], [])

            UV_seen_hois = np.setdiff1d(valid_hois, hico_unseen_index["unseen_verb"])
            UV_unseen_hois = np.setdiff1d(hico_unseen_index["unseen_verb"], [])

            UO_seen_hois = np.setdiff1d(valid_hois, hico_unseen_index["unseen_object"])
            UO_unseen_hois = np.setdiff1d(hico_unseen_index["unseen_object"], [])

            NF_seen_hois = np.setdiff1d(valid_hois, hico_unseen_index["non_rare_first"])
            NF_unseen_hois = np.setdiff1d(hico_unseen_index["non_rare_first"], [])

        else:
            valid_hois = np.setdiff1d(np.arange(600), [])
            seen_hois = np.setdiff1d(valid_hois, self.zero_shot_interaction_ids)
            zero_shot_hois = np.setdiff1d(self.zero_shot_interaction_ids, [])

            rare_hois = np.setdiff1d(RARE_INTERACTION_IDS, [])
            non_rare_hois = np.setdiff1d(valid_hois, RARE_INTERACTION_IDS)

            RF_seen_hois = np.setdiff1d(valid_hois, hico_unseen_index["rare_first"])
            RF_unseen_hois = np.setdiff1d(hico_unseen_index["rare_first"], [])

            UV_seen_hois = np.setdiff1d(valid_hois, hico_unseen_index["unseen_verb"])
            UV_unseen_hois = np.setdiff1d(hico_unseen_index["unseen_verb"], [])

            UO_seen_hois = np.setdiff1d(valid_hois, hico_unseen_index["unseen_object"])
            UO_unseen_hois = np.setdiff1d(hico_unseen_index["unseen_object"], [])

            NF_seen_hois = np.setdiff1d(valid_hois, hico_unseen_index["non_rare_first"])
            NF_unseen_hois = np.setdiff1d(hico_unseen_index["non_rare_first"], [])

        full_mAP = np.mean(self.hico_ap[valid_hois])
        zero_shot_mAP = np.mean(self.hico_ap[zero_shot_hois])

        rare_mAP = np.mean(self.hico_ap[rare_hois])
        non_rare_mAP = np.mean(self.hico_ap[non_rare_hois])
        RF_seen_mAP = np.mean(self.hico_ap[RF_seen_hois])
        RF_unseen_mAP = np.mean(self.hico_ap[RF_unseen_hois])
        UV_seen_mAP = np.mean(self.hico_ap[UV_seen_hois])
        UV_unseen_mAP = np.mean(self.hico_ap[UV_unseen_hois])
        UO_seen_mAP = np.mean(self.hico_ap[UO_seen_hois])
        UO_unseen_mAP = np.mean(self.hico_ap[UO_unseen_hois])
        NF_seen_mAP = np.mean(self.hico_ap[NF_seen_hois])
        NF_unseen_mAP = np.mean(self.hico_ap[NF_unseen_hois])

        print("zero-shot mAP: {:.2f}".format(zero_shot_mAP * 100.))
        print("full mAP: {:.2f}".format(full_mAP * 100.))
        print("rare mAP: {:.2f}".format(rare_mAP * 100.))
        print("non-rare mAP: {:.2f}".format(non_rare_mAP * 100.))
        print("RF_seen mAP: {:.2f}".format(RF_seen_mAP * 100.))
        print("RF_unseen mAP: {:.2f}".format(RF_unseen_mAP * 100.))
        print("UV_seen mAP: {:.2f}".format(UV_seen_mAP * 100.))
        print("UV_unseen mAP: {:.2f}".format(UV_unseen_mAP * 100.))
        print("UO_seen mAP: {:.2f}".format(UO_seen_mAP * 100.))
        print("UO_unseen mAP: {:.2f}".format(UO_unseen_mAP * 100.))
        print("NF_seen mAP: {:.2f}".format(NF_seen_mAP * 100.))
        print("NF_unseen mAP: {:.2f}".format(NF_unseen_mAP * 100.))

        print("\n========== RF/NF unseen mAP with / without no_interaction ==========")
        self._print_unseen_map_with_without_no_interaction("RF_unseen", RF_unseen_hois)
        self._print_unseen_map_with_without_no_interaction("NF_unseen", NF_unseen_hois)

        self._print_no_interaction_missing_stats()

        # Diagnostics for investigating low NF_unseen performance
        print("\n========== Detailed diagnostics ==========")
        self._print_split_diagnostics("NF_seen", NF_seen_hois)
        self._print_split_diagnostics("NF_unseen", NF_unseen_hois)
        self._print_nf_unseen_bottomk(NF_unseen_hois, topk=30)

    def save_preds(self):
        with open(os.path.join(self.output_dir, "preds.pkl"), "wb") as f:
            pickle.dump({"scores": self.scores, "boxes": self.boxes, "keys": self.keys}, f)

    def save(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        with open(os.path.join(output_dir, "dets.pkl"), "wb") as f:
            pickle.dump({"gts": self.gts, "scores": self.scores, "boxes": self.boxes, "keys": self.keys}, f)

    def _compute_no_interaction_missing_stats(self, dataset_dicts):
        """
        Count missing no_interaction labels.

        Definition:
        For each image, enumerate every person(subject)-object pair.
        If a pair has no HOI annotation at all, then this pair should have a
        no_interaction annotation and is counted as missing.
        """
        object_name2id = {x["name"]: x["id"] for x in HICO_OBJECTS}
        action_id2name = {x["id"]: x["name"] for x in HICO_ACTIONS}
        object_id2name = {x["id"]: x["name"] for x in HICO_OBJECTS}
        hoi_mapper = {(x["action"], x["object"]): x["interaction_id"] for x in HICO_INTERACTIONS}

        person_category_id = object_name2id.get("person", 1)

        total_hoi_annotations = 0
        max_hoi_annotations_per_image = 0
        image_ids_with_max_hoi_annotations = []

        total_candidate_pairs = 0
        total_annotated_pairs = 0
        total_existing_no_interaction_annotations = 0
        total_existing_no_interaction_pairs = 0
        total_missing_no_interaction_pairs = 0

        per_image_hoi_annotations = []
        per_image_candidate_pairs = []
        per_image_missing_pairs = []
        per_image_missing_ratios = []

        for anno_dict in dataset_dicts:
            image_id = anno_dict.get("img_id", "unknown")
            box_annos = anno_dict.get("annotations", [])
            hoi_annos = anno_dict.get("hoi_annotation", [])

            num_hoi_annotations = len(hoi_annos)
            total_hoi_annotations += num_hoi_annotations
            per_image_hoi_annotations.append(num_hoi_annotations)

            if num_hoi_annotations > max_hoi_annotations_per_image:
                max_hoi_annotations_per_image = num_hoi_annotations
                image_ids_with_max_hoi_annotations = [image_id]
            elif num_hoi_annotations == max_hoi_annotations_per_image:
                image_ids_with_max_hoi_annotations.append(image_id)

            subject_indices = [
                idx for idx, anno in enumerate(box_annos)
                if anno.get("category_id") == person_category_id
            ]
            object_indices = list(range(len(box_annos)))

            candidate_pairs = set()
            for subject_id in subject_indices:
                for object_id in object_indices:
                    if subject_id == object_id:
                        continue
                    candidate_pairs.add((subject_id, object_id))

            annotated_pairs = set()
            existing_no_interaction_pairs = set()

            for hoi in hoi_annos:
                subject_id = hoi["subject_id"]
                object_id = hoi["object_id"]
                pair = (subject_id, object_id)
                annotated_pairs.add(pair)

                # category_id in annotation starts from 1.
                action_id = hoi["category_id"] - 1
                object_category_id = box_annos[object_id]["category_id"]

                action_name = action_id2name[action_id]
                object_name = object_id2name[object_category_id]
                hoi_id = hoi_mapper[(action_name, object_name)]

                if hoi_id in NON_INTERACTION_IDS:
                    total_existing_no_interaction_annotations += 1
                    existing_no_interaction_pairs.add(pair)

            annotated_candidate_pairs = annotated_pairs.intersection(candidate_pairs)
            missing_no_interaction_pairs = candidate_pairs.difference(annotated_candidate_pairs)

            num_candidate = len(candidate_pairs)
            num_missing = len(missing_no_interaction_pairs)

            total_candidate_pairs += num_candidate
            total_annotated_pairs += len(annotated_candidate_pairs)
            total_existing_no_interaction_pairs += len(existing_no_interaction_pairs.intersection(candidate_pairs))
            total_missing_no_interaction_pairs += num_missing

            per_image_candidate_pairs.append(num_candidate)
            per_image_missing_pairs.append(num_missing)
            if num_candidate > 0:
                per_image_missing_ratios.append(num_missing / float(num_candidate))
            else:
                per_image_missing_ratios.append(0.0)

        num_images = len(dataset_dicts)
        dataset_missing_ratio = (
            total_missing_no_interaction_pairs / float(total_candidate_pairs)
            if total_candidate_pairs > 0 else 0.0
        )

        return {
            "num_images": num_images,
            "total_hoi_annotations": int(total_hoi_annotations),
            "avg_hoi_annotations_per_image": float(np.mean(per_image_hoi_annotations)) if num_images > 0 else 0.0,
            "max_hoi_annotations_per_image": int(max_hoi_annotations_per_image),
            "image_ids_with_max_hoi_annotations": image_ids_with_max_hoi_annotations,
            "total_candidate_pairs": int(total_candidate_pairs),
            "total_annotated_pairs": int(total_annotated_pairs),
            "total_existing_no_interaction_annotations": int(total_existing_no_interaction_annotations),
            "total_existing_no_interaction_pairs": int(total_existing_no_interaction_pairs),
            "total_missing_no_interaction_pairs": int(total_missing_no_interaction_pairs),
            "avg_missing_no_interaction_pairs_per_image": (
                float(np.mean(per_image_missing_pairs)) if num_images > 0 else 0.0
            ),
            "avg_candidate_pairs_per_image": (
                float(np.mean(per_image_candidate_pairs)) if num_images > 0 else 0.0
            ),
            "dataset_missing_no_interaction_ratio": float(dataset_missing_ratio),
            "mean_per_image_missing_no_interaction_ratio": (
                float(np.mean(per_image_missing_ratios)) if num_images > 0 else 0.0
            ),
        }

    def load_anno(self, anno_file):
        with open(anno_file, "r") as f:
            dataset_dicts = json.load(f)

        self.no_interaction_missing_stats = self._compute_no_interaction_missing_stats(dataset_dicts)

        action_id2name = {x["id"]: x["name"] for x in HICO_ACTIONS}
        object_id2name = {x["id"]: x["name"] for x in HICO_OBJECTS}
        hoi_mapper = {(x["action"], x["object"]): x["interaction_id"] for x in HICO_INTERACTIONS}

        size = self.size
        gts = {i: collections.defaultdict(list) for i in range(size)}
        for anno_dict in dataset_dicts:
            image_id = anno_dict["img_id"]
            box_annos = anno_dict.get("annotations", [])
            hoi_annos = anno_dict.get("hoi_annotation", [])
            for hoi in hoi_annos:
                person_box = box_annos[hoi["subject_id"]]["bbox"]
                object_box = box_annos[hoi["object_id"]]["bbox"]
                action_id = hoi["category_id"] - 1  # original annotations start from 1
                object_id = box_annos[hoi["object_id"]]["category_id"]  # original annotations start from 1
                hoi_id = hoi_mapper[(action_id2name[action_id], object_id2name[object_id])]
                gts[hoi_id][image_id].append(person_box + object_box)

        for hoi_id in gts:
            for img_id in gts[hoi_id]:
                gts[hoi_id][img_id] = np.array(gts[hoi_id][img_id])

        return gts


def calc_ap(scores, boxes, keys, gt_boxes):
    if len(keys) == 0:
        return 0, 0

    if isinstance(boxes, list):
        scores, boxes, keys = np.array(scores), np.array(boxes), np.array(keys)

    hit = []
    idx = np.argsort(scores)[::-1]
    npos = 0
    used = {}

    for key in gt_boxes.keys():
        npos += gt_boxes[key].shape[0]
        used[key] = set()

    for i in range(min(len(idx), 39999)):
        pair_id = idx[i]
        box = boxes[pair_id, :]
        key = keys[pair_id]
        if key in gt_boxes:
            maxi = 0.0
            k = -1
            for j in range(gt_boxes[key].shape[0]):
                tmp = calc_hit(box, gt_boxes[key][j, :])
                if maxi < tmp:
                    maxi = tmp
                    k = j
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)

    bottom = np.array(range(len(hit))) + 1
    hit = np.cumsum(hit)
    rec = hit / npos if npos > 0 else hit / (npos + 1e-8)
    prec = hit / bottom

    ap = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec) if len(rec) else 0


def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)


def iou(bb1, bb2, debug=False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)

    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)