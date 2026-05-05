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

        y_true = []
        y_scores = []

        for det in detected:
            best_iou = 0
            best_gt_idx = -1

            for i, gt_box in enumerate(gt):
                if i in matched_gt:
                    continue

                iou = Metrics.iou(det, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            # Score SIEMPRE se guarda
            score = getattr(det, "score", 1.0)

            if best_iou > iou_threshold:
                TP += 1
                matched_gt.add(best_gt_idx)
                y_true.append(1)
            else:
                FP += 1
                y_true.append(0)

            y_scores.append(score)

        FN = len(gt) - len(matched_gt)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        return {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'precision': precision,
            'recall': recall,
            'y_true': y_true,
            'y_scores': y_scores
        }