from src.bounding_box import BoundingBox
from src.image_loader import ImageLoader
from src.preprocessor import Preprocessor
from src.cell_detector import CellDetector
from src.counter import Counter
from src.xml_generator import XMLGenerator
from src.visualizer import Visualizer
from src.storage_manager import StorageManager
from src.logger import Logger
from src.annotation_reader import AnnotationReader
from src.metrics import evaluate_detection
from typing import List

class RBCPipeline:

    def __init__(self) -> None:
        self.loader = ImageLoader()
        self.preprocessor = Preprocessor()
        self.detector = CellDetector()
        self.counter = Counter()
        self.xml_generator = XMLGenerator()
        self.visualizer = Visualizer()
        self.storage = StorageManager()
        self.logger = Logger()

    def procesar_imagen(self, ruta_imagen: str, ruta_salida: str) -> None:

        try:
            self.logger.info(f"Iniciando procesamiento: {ruta_imagen}")
            # 1. Cargar y preprocesar imagen
            imagen = self.loader.load_image(ruta_imagen)
            imagen_pre = self.preprocessor.process(imagen)

            # 2. Detectar células
            bboxes = self.detector.detect(imagen_pre)

            total = self.counter.contar(bboxes)

            xml = self.xml_generator.generate_xml(ruta_imagen, bboxes)

            overlay = self.visualizer.draw_bboxes(imagen, bboxes)

            nombre_base = ruta_imagen.split("/")[-1].split(".")[0]

            self.storage.save_results(
                imagen, xml, [(overlay, "detected")], ruta_salida, nombre_base
            )

            self.logger.info(f"Procesamiento completado: {total} células detectadas")

        except Exception as e:
            self.logger.error(f"Error procesando {ruta_imagen}: {str(e)}")
            raise

    def procesar_con_gt(self, ruta_imagen: str, ruta_xml_gt: str, ruta_salida: str) -> dict:
        """
        Procesa la imagen, detecta células y compara con las bboxes del XML (ground truth)
        usando AnnotationReader.
        Devuelve métricas y genera overlay visual.
        """
        try:
            self.logger.info(f"Iniciando procesamiento con GT: {ruta_imagen}")

            # 1. Cargar y preprocesar imagen
            imagen = self.loader.load_image(ruta_imagen)
            imagen_pre = self.preprocessor.process(imagen)

            # 2. Detectar células
            bboxes_detectadas: List[BoundingBox] = self.detector.detect(imagen_pre)

            # 3. Leer ground truth desde XML usando AnnotationReader
            bboxes_gt: List[BoundingBox] = AnnotationReader.from_xml(ruta_xml_gt)

            # 4. Evaluar métricas
            metrics = evaluate_detection(bboxes_detectadas, bboxes_gt)

            # 5. Crear overlays para visualización
            # Verde = Ground Truth, Rojo = Detectado
            ground_truth_overlay = self.visualizer.draw_bboxes(imagen, bboxes_gt, color=(0, 255, 0))
            detected_overlay = self.visualizer.draw_bboxes(imagen, bboxes_detectadas, color=(0, 0, 255))

            # 6. Guardar resultados
            nombre_base = ruta_imagen.split("/")[-1].split(".")[0]
            xml_generado = self.xml_generator.generate_xml(ruta_imagen, bboxes_detectadas)
            self.storage.save_results(imagen, xml_generado, [(ground_truth_overlay, "gt"), (detected_overlay, "detected")], ruta_salida, nombre_base)

            self.logger.info(
                f"Procesamiento con GT completado: {len(bboxes_detectadas)} detectadas, "
                f"Precisión: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}"
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error procesando con GT {ruta_imagen}: {str(e)}")
            raise