from typing import List
from src.bounding_box import BoundingBox

class Counter:

    def contar(self, bboxes: List[BoundingBox]) -> int:
        """
        Cuenta el número de células detectadas

        :param bboxes: Lista de bounding boxes
        :return: Número total
        """
        return len(bboxes)