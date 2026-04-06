import cv2
import numpy as np
from typing import List
from src.bounding_box import BoundingBox

class CellDetector:

    def detect(self, imagen: np.ndarray) -> List[BoundingBox]:
        """
        Detecta glóbulos rojos (versión simplificada)

        :param imagen: Imagen preprocesada
        :return: Lista de bounding boxes
        """
        # 1. Suavizado
        blur = cv2.GaussianBlur(imagen, (7, 7), 0)

        # 2. Detección de bordes
        edges = cv2.Canny(blur, 50, 150)

        # 3. Dilatar para unir bordes
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 4. Contornos
        contornos, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes: List[BoundingBox] = []

        for cnt in contornos:
            x, y, w, h = cv2.boundingRect(cnt)

            area = w * h

            #FILTRO CLAVE (ajústar)
            if 50 < area < 5000:
                bboxes.append(BoundingBox(x, y, w, h))

        return bboxes