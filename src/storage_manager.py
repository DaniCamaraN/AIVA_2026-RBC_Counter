import cv2
import os
from typing import List, Tuple
import numpy as np
from src.bounding_box import BoundingBox

class StorageManager:
    """
    Gestiona el guardado de imágenes, overlays y XML.
    Permite guardar varias versiones de overlays (detecciones, GT, comparaciones).
    """

    def save_results(
        self,
        imagen: np.ndarray,
        xml: str,
        overlays: List[Tuple[np.ndarray, str]] = None,
        ruta_salida: str = "output",
        nombre_base: str = "result"
    ) -> None:
        """
        Guarda imagen original, XML y uno o varios overlays.

        :param imagen: Imagen original
        :param xml: XML generado de detecciones
        :param overlays: Lista de tuplas (imagen_overlay, sufijo_nombre)
        :param ruta_salida: Carpeta destino
        :param nombre_base: Nombre base de archivo
        """
        os.makedirs(ruta_salida, exist_ok=True)

        # Guardar imagen original y XML
        cv2.imwrite(f"{ruta_salida}/{nombre_base}.jpg", imagen)
        with open(f"{ruta_salida}/{nombre_base}.xml", "w") as f:
            f.write(xml)

        # Guardar todos los overlays
        if overlays:
            for overlay_img, sufijo in overlays:
                cv2.imwrite(f"{ruta_salida}/{nombre_base}_{sufijo}.jpg", overlay_img)