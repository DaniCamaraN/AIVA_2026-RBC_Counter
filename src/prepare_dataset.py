from src.xml_to_yolo_converter import XMLToYOLOConverter
from src.dataset_splitter import DatasetSplitter


class DatasetPreparationPipeline:

    def __init__(self) -> None:
        self.converter = XMLToYOLOConverter(
            xml_dir="data/Annotations",
            output_dir="data/labels"
        )

        self.splitter = DatasetSplitter(
            images_dir="data/JPEGImages",
            labels_dir="data/labels",
            annotations_dir="data/Annotations",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

    def run(self):
        print("Paso 1: Convertir XML a YOLO")
        self.converter.run()

        print("Paso 2: Dividir dataset")
        self.splitter.run()

        print("Dataset preparado correctamente")