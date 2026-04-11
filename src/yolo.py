from pathlib import Path
from typing import Optional
from ultralytics import YOLO

class YOLOTrainer:
    def __init__(
        self,
        data_yaml: str,
        base_model: str = "yolo11n.pt",
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 8,
        project: str = "./models",
        name: str = "rbc_train",
        device: Optional[str] = None,
    ) -> None:
        """
        YOLO para detección de glóbulos rojos.

        :param data_yaml: Ruta al archivo rbc.yaml
        :param base_model: Modelo base preentrenado
        :param epochs: Número de épocas
        :param imgsz: Tamaño de imagen
        :param batch: Batch size
        :param project: Carpeta base de resultados
        :param name: Nombre del experimento
        :param device: "cpu", "0", "0,1", etc. Si es None, YOLO decide
        """
        self.data_yaml = data_yaml
        self.base_model = base_model
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.project = project
        self.name = name
        self.device = device

        self.model = YOLO(self.base_model)

    def train(self):
        results = self.model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            project=self.project,
            name=self.name,
            device=self.device,
        )
        return results

    def validate(self):
        return self.model.val()

    def get_best_model_path(self) -> str:
        best_path = Path(self.project) / self.name / "best.pt"
        return str(best_path)

    def load_best_model(self) -> YOLO:
        best_path = self.get_best_model_path()
        return YOLO(best_path)