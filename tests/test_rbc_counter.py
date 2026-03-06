import unittest
import numpy as np
import cv2
import os
from src.main import cargar_imagen

class TestCargarImagen(unittest.TestCase):

    def test_imagen_inexistente(self):
        """Comprobar que lanza error si la imagen no existe"""
        with self.assertRaises(FileNotFoundError):
            cargar_imagen("imagen_que_no_existe.jpg")

    def test_imagen_valida(self):
        """Comprobar que carga una imagen válida"""
        # Creamos una imagen temporal negra
        nombre_temp = "temp.jpg"
        cv2.imwrite(nombre_temp, np.zeros((100, 100, 3), dtype=np.uint8))

        imagen = cargar_imagen(nombre_temp)
        self.assertEqual(imagen.shape, (100, 100, 3))

        # Borramos la imagen temporal
        os.remove(nombre_temp)

if __name__ == "__main__":
    unittest.main()