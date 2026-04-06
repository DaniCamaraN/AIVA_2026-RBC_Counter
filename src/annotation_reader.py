import xml.etree.ElementTree as ET
from typing import List
from src.bounding_box import BoundingBox

class AnnotationReader:
    """Lee un XML de anotaciones y devuelve las bounding boxes (ground truth)."""

    @staticmethod
    def from_xml(path: str) -> List[BoundingBox]:
        tree = ET.parse(path)
        root = tree.getroot()
        bboxes: List[BoundingBox] = []

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            bboxes.append(BoundingBox(xmin, ymin, xmax - xmin, ymax - ymin))

        return bboxes