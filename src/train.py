from src.xml_to_yolo_converter import XMLToYOLOConverter
from src.dataset_splitter import DatasetSplitter
from src.yolo import YOLOTrainer

class TrainingPipeline:

    def __init__(self) -> None:
        self.converter = XMLToYOLOConverter(
            xml_dir="data/Annotations",
            output_dir="data/labels"
        )

        self.splitter = DatasetSplitter(
            images_dir="data/JPEGImages",
            labels_dir="data/labels",
            output_base="dataset_rbc",
            train_ratio=0.9
        )

        self.trainer = YOLOTrainer(
            data_yaml="dataset_rbc/rbc.yaml",
            base_model="yolo11n.pt",
            epochs=20,
            imgsz=640,
            batch=8,
            project="models",
            name="rbc_train"
        )

    def run(self):
        print("Paso 1: Convertir XML a YOLO")
        self.converter.run()

        print("Paso 2: Dividir dataset")
        self.splitter.run()

        print("Paso 3: Entrenar modelo YOLO")
        self.trainer.train()

        print("Entrenamiento completo")
        print("Modelo:", self.trainer.get_best_model_path())