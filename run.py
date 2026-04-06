from src.pipeline import RBCPipeline

DATASET_FOLDER = "data/JPEGImages"  # Ruta al dataset de imágenes

if __name__ == "__main__":
    pipeline = RBCPipeline()

    ruta_imagen = f"{DATASET_FOLDER}/BloodImage_00000.jpg"
    ruta_salida = "output"

    pipeline.procesar_imagen(ruta_imagen, ruta_salida)