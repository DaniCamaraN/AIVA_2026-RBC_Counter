import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

class XMLToYOLOConverter:

    def __init__(
        self,
        xml_dir: str,
        output_dir: str,
        class_map: Dict[str, int] = None
    ) -> None:
        """
        :param xml_dir: Carpeta con anotaciones XML
        :param output_dir: Carpeta de salida para labels YOLO
        :param class_map: Mapeo de clases (ej: {"RBC": 0})
        """
        self.xml_dir = Path(xml_dir)
        self.output_dir = Path(output_dir)
        self.class_map = class_map or {"RBC": 0}

    def convert_bbox(self, size, xmin, ymin, xmax, ymax):
        w_img, h_img = size

        x_center = ((xmin + xmax) / 2.0) / w_img
        y_center = ((ymin + ymax) / 2.0) / h_img
        w = (xmax - xmin) / w_img
        h = (ymax - ymin) / h_img

        return x_center, y_center, w, h

    def convert_file(self, xml_path: Path, txt_path: Path) -> None:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w_img = int(size.find("width").text)
        h_img = int(size.find("height").text)

        lines = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()

            if class_name not in self.class_map:
                continue

            class_id = self.class_map[class_name]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x_center, y_center, w, h = self.convert_bbox(
                (w_img, h_img), xmin, ymin, xmax, ymax
            )

            line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            lines.append(line)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def run(self) -> None:
        print("Convirtiendo XML a formato YOLO...")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for xml_file in self.xml_dir.iterdir():
            if xml_file.suffix != ".xml":
                continue

            txt_name = xml_file.stem + ".txt"
            txt_path = self.output_dir / txt_name

            self.convert_file(xml_file, txt_path)

            print(f"Convertido {xml_file.name} -> {txt_name}")
