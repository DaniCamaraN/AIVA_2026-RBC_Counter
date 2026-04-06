import xml.etree.ElementTree as ET
from typing import List
from src.bounding_box import BoundingBox

class XMLGenerator:

    def generate_xml(self, nombre_imagen: str, bboxes: List[BoundingBox]) -> str:
        """
        Genera XML con las detecciones

        :param nombre_imagen: Nombre de la imagen
        :param bboxes: Lista de bounding boxes
        :return: String XML
        """
        root = ET.Element("imagen", nombre=nombre_imagen)

        for bbox in bboxes:
            ET.SubElement(
                root,
                "bbox",
                x=str(bbox.x),
                y=str(bbox.y),
                width=str(bbox.width),
                height=str(bbox.height),
            )

        xml_str: str = ET.tostring(root, encoding="unicode")
        return xml_str