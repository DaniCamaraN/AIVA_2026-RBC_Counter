from pathlib import Path
from src.pipeline import RBCPipeline

DATASET_FOLDER = "dataset_rbc/images/test"
OUTPUT_FOLDER = "output"

if __name__ == "__main__":
    pipeline = RBCPipeline()

    dataset_path = Path(DATASET_FOLDER)

    for img_path in dataset_path.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        nombre_base = img_path.stem

        ruta_salida = Path(OUTPUT_FOLDER) / nombre_base

        print(f"Procesando: {img_path.name}")

        pipeline.procesar_imagen(str(img_path), str(ruta_salida))
