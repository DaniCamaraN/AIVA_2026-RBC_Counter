import cv2
import numpy as np
from typing import List
from src.bounding_box import BoundingBox

class Visualizer:

    def draw_bboxes(
        self, imagen: np.ndarray, bboxes: List[BoundingBox], color: tuple = (0, 255, 0)
    ) -> np.ndarray:
        """
        Dibuja bounding boxes sobre la imagen

        :param imagen: Imagen original
        :param bboxes: Lista de bounding boxes
        :return: Imagen con overlay
        """
        output: np.ndarray = imagen.copy()

        for bbox in bboxes:
            cv2.rectangle(
                output,
                (bbox.x, bbox.y),
                (bbox.x + bbox.width, bbox.y + bbox.height),
                color,
                2,
            )

        return output