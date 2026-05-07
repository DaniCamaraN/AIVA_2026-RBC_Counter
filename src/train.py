from src.xml_to_yolo_converter import XMLToYOLOConverter
from src.dataset_splitter import DatasetSplitter
from src.yolo import YOLOTrainer

class TrainingPipeline:

    def __init__(self) -> None:
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
        print("Entrenando modelo YOLO")
        self.trainer.train()

        print("Entrenamiento completo")
        print("Modelo:", self.trainer.get_best_model_path())