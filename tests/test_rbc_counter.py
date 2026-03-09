import unittest
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from src.main import cargar_imagen

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

if __name__ == "__main__":
    unittest.main()