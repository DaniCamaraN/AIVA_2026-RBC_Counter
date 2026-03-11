import cv2
import xml.etree.ElementTree as ET

def dibujar_bboxes_desde_xml(image_path, xml_path, output_path):
    """
    Lee una imagen y su archivo XML asociado con bounding boxes,
    dibuja las cajas sobre la imagen y guarda el resultado.

    Args:
        image_path (str): Ruta a la imagen.
        xml_path (str): Ruta al archivo XML con las bboxes.
        output_path (str): Ruta donde guardar la imagen resultante.
    """

    # Leer imagen
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    # Parsear XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    count = 0

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Dibujar rectángulo
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        count += 1

    # Añadir conteo
    cv2.putText(
        img,
        f"Globulos rojos: {count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Guardar resultado
    cv2.imwrite(output_path, img)

    print(f"Mockup guardado en: {output_path}")
    return count