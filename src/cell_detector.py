import cv2
import numpy as np
from typing import List, Optional
from ultralytics import YOLO
from src.bounding_box import BoundingBox

class CellDetector:

    def __init__(
        self,
        model_path: str = "runs/detect/models/rbc_train/weights/best.pt",
        conf_threshold: float = 0.65,
        class_id: Optional[int] = 0
    ) -> None:
        """
        :param model_path: Ruta al modelo entrenado (best.pt)
        :param conf_threshold: Umbral de confianza
        :param class_id: Clase a detectar (0 = RBC)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_id = class_id

    def detect(self, imagen: np.ndarray) -> List[BoundingBox]:
        """
        Detecta glóbulos rojos usando YOLO.

        :param imagen: Imagen original (NO preprocesada)
        :return: Lista de bounding boxes
        """

        # Asegurar 3 canales (por si viene en gris)
        if imagen.ndim == 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        elif imagen.ndim == 3 and imagen.shape[2] == 1:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

        results = self.model.predict(
            source=imagen,
            conf=self.conf_threshold,
            verbose=False
        )

        bboxes: List[BoundingBox] = []

        if not results:
            return bboxes

        result = results[0]

        if result.boxes is None:
            return bboxes

        for box in result.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())

            # Filtrar clase (RBC = 0)
            if self.class_id is not None and cls != self.class_id:
                continue

            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x = int(round(x1))
            y = int(round(y1))
            w = int(round(x2 - x1))
            h = int(round(y2 - y1))

            if w > 0 and h > 0:
                bboxes.append(BoundingBox(x, y, w, h))

        return bboxes