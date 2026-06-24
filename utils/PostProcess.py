from torchvision.ops import batched_nms
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import utils.box_ops as box_ops

class PostProcess(object):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, score_threshold, bbox_lambda=1, enable_softmax=False):
        self.score_threshold = score_threshold
        self.bbox_lambda = bbox_lambda
        self.enable_softmax = enable_softmax


    def __call__(self, outputs, hoi_mapper):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            original_size: For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            hoi_mapper: map the predicted classes to the hoi id specified by the dataset.
        """
        # Recover the bounding boxes based on the original image size
        pred_boxes = outputs['pred_boxes']
        pred_person_boxes = pred_boxes[:, :4]
        pred_object_boxes = pred_boxes[:, 4:]


        if self.enable_softmax:
            hoi_scores = outputs['pred_logits'].softmax(dim=-1)
        else:
            hoi_scores = outputs['pred_logits'].sigmoid()
        box_scores = outputs['box_scores'].sigmoid()
        

        scores = hoi_scores * (box_scores ** self.bbox_lambda)


        keep = torch.nonzero(scores > self.score_threshold, as_tuple=True)

        scores = scores[keep].float()
        classes = keep[1] # 哪些HOI类是合适的 第0维是HOI的个数， 第1维是HOI的类别
        pred_person_boxes = pred_person_boxes[keep[0]].float()
        pred_object_boxes = pred_object_boxes[keep[0]].float()

        person_keep = batched_nms(pred_person_boxes, scores, classes, 0.5)
        object_keep = batched_nms(pred_object_boxes, scores, classes, 0.5)

        person_filter_mask = torch.zeros_like(scores, dtype=torch.bool)
        object_filter_mask = torch.zeros_like(scores, dtype=torch.bool)
        person_filter_mask[person_keep] = True
        object_filter_mask[object_keep] = True
        filter_mask = torch.logical_or(person_filter_mask, object_filter_mask)
        # torch.logical_or 这是 PyTorch 库中的一个函数，用于执行逻辑或操作。

        scores = scores[filter_mask].detach().cpu().numpy().tolist()
        classes = classes[filter_mask].detach().cpu().numpy().tolist()
        pred_boxes = torch.cat([pred_person_boxes, pred_object_boxes], dim=-1)
        pred_boxes = pred_boxes[filter_mask].detach().cpu().numpy().tolist()

        results = []
        for score, hoi_id, boxes in zip(scores, classes, pred_boxes):
            results.append([hoi_mapper[int(hoi_id)], score] + boxes)
            # results.append([int(hoi_id), score] + boxes)
        return results
