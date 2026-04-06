from src.pipeline import RBCPipeline

DATASET_FOLDER = "data/JPEGImages"  # Carpeta con las imágenes
ANNOTATIONS_FOLDER = "data/Annotations"  # Carpeta con los XML ground truth
OUTPUT_FOLDER = "output"  # Carpeta donde se guardarán los resultados

if __name__ == "__main__":
    pipeline = RBCPipeline()

    # Imagen de prueba
    nombre_imagen = "BloodImage_00000"
    ruta_imagen = f"{DATASET_FOLDER}/{nombre_imagen}.jpg"
    ruta_xml_gt = f"{ANNOTATIONS_FOLDER}/{nombre_imagen}.xml"

    # Ejecutar pipeline con comparación a ground truth
    metrics = pipeline.procesar_con_gt(ruta_imagen, ruta_xml_gt, OUTPUT_FOLDER)

    print("Métricas obtenidas:")
    print(f"Precisión: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")