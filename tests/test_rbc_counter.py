import unittest
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from mockup.mockup import cargar_imagen

DATASET_FOLDER = "data/JPEGImages"  # Ruta al dataset de imágenes

# Funciones simuladas para el test
def detectar_globulos(imagen):
    """Mock simple de detección: devuelve lista de bboxes simuladas"""
    return [(10,10,20,20), (30,30,40,40)]  # dos "células" de ejemplo

def generar_xml(nombre_imagen, bboxes, output_folder="output"):
    """Genera un XML con las bboxes simuladas"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    root = ET.Element("detecciones")
    for i, bbox in enumerate(bboxes):
        cel = ET.SubElement(root, "celula", id=str(i))
        cel.set("x1", str(bbox[0]))
        cel.set("y1", str(bbox[1]))
        cel.set("x2", str(bbox[2]))
        cel.set("y2", str(bbox[3]))
    tree = ET.ElementTree(root)
    xml_path = os.path.join(output_folder, f"{nombre_imagen}.xml")
    tree.write(xml_path)
    return xml_path

class TestRBC_Counter(unittest.TestCase):

    def test_imagen_inexistente(self):
        """Comprueba que cargar una imagen inexistente lanza error"""
        with self.assertRaises(FileNotFoundError):
            cargar_imagen("imagen_que_no_existe.jpg")

    def test_imagen_valida(self):
        """Comprueba que se carga correctamente una imagen válida"""
        nombre_temp = "temp.jpg"
        cv2.imwrite(nombre_temp, np.zeros((100,100,3), dtype=np.uint8))
        imagen = cargar_imagen(nombre_temp)
        self.assertEqual(imagen.shape, (100,100,3))
        os.remove(nombre_temp)

    def test_deteccion_globulos(self):
        """Comprueba que la función de detección devuelve lista de bboxes"""
        imagen = np.zeros((100,100,3), dtype=np.uint8)
        bboxes = detectar_globulos(imagen)
        self.assertIsInstance(bboxes, list)
        self.assertTrue(all(len(b) == 4 for b in bboxes))

    def test_generacion_xml(self):
        """Comprueba que se genera un XML correcto con las bboxes"""
        nombre_imagen = "test_image"
        bboxes = [(0,0,10,10), (10,10,20,20)]
        xml_path = generar_xml(nombre_imagen, bboxes, output_folder="test_output")
        self.assertTrue(os.path.exists(xml_path))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        self.assertEqual(len(root.findall("celula")), len(bboxes))
        # Limpiar
        os.remove(xml_path)
        os.rmdir("test_output")

    def test_bboxes_validas(self):
        """Comprueba que las bboxes generadas tienen coordenadas válidas y dentro de la imagen"""
        imagen = np.zeros((100,100,3), dtype=np.uint8)
        bboxes = detectar_globulos(imagen)

        for x1, y1, x2, y2 in bboxes:
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
    def test_dataset_no_vacio(self):
        self.assertTrue(len(os.listdir(DATASET_FOLDER)) > 0,
                        "El dataset está vacío")

    def test_resolucion_dataset(self):
        """
        Comprueba que todas las imágenes del dataset tienen resolución 640x480
        """
        contador_imagenes = 0
        for filename in os.listdir(DATASET_FOLDER):

            if filename.lower().endswith((".jpg", ".jpeg", ".png")):

                path = os.path.join(DATASET_FOLDER, filename)
                imagen = cargar_imagen(path)
                contador_imagenes += 1

                self.assertIsNotNone(imagen, f"No se pudo cargar {filename}")

                h, w = imagen.shape[:2]

                self.assertEqual(
                    (w, h),
                    (640, 480),
                    f"La imagen {filename} tiene resolución {w}x{h}, debería ser 640x480"
                )
        self.assertGreater(contador_imagenes, 0, "No se encontraron imágenes en el dataset")
        self.assertEqual(contador_imagenes, len(os.listdir(DATASET_FOLDER)),
                         "No todas las entradas del dataset son imágenes válidas")

if __name__ == "__main__":
    unittest.main()
