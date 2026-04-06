import cv2
import numpy as np

class Preprocessor:
    def process(self, imagen: np.ndarray) -> np.ndarray:
        # Convertir a escala de grises
        grey: np.ndarray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Suavizado para eliminar ruido
        blur: np.ndarray = cv2.GaussianBlur(grey, (5, 5), 0)

        return blur