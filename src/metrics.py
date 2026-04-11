from typing import List
from src.bounding_box import BoundingBox

class Metrics:

    @staticmethod
    def iou(box1: BoundingBox, box2: BoundingBox) -> float:
        xA = max(box1.x, box2.x)
        yA = max(box1.y, box2.y)
        xB = min(box1.x + box1.w, box2.x + box2.w)
        yB = min(box1.y + box1.h, box2.y + box2.h)
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = box1.w * box1.h
        box2Area = box2.w * box2.h
        
        return interArea / float(box1Area + box2Area - interArea) if (box1Area + box2Area - interArea) > 0 else 0

    @staticmethod
    def evaluate_detection(detected: List[BoundingBox], gt: List[BoundingBox], iou_threshold: float = 0.5):
        TP = 0
        FP = 0
        matched_gt = set()

        for det in detected:
            match_found = False
            for i, gt_box in enumerate(gt):
                if i in matched_gt:
                    continue
                if Metrics.iou(det, gt_box) > iou_threshold:
                    TP += 1
                    matched_gt.add(i)
                    match_found = True
                    break
            if not match_found:
                FP += 1

        FN = len(gt) - len(matched_gt)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        return {'TP': TP, 'FP': FP, 'FN': FN, 'precision': precision, 'recall': recall}