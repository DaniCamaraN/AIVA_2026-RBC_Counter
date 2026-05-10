import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, auc
from pathlib import Path
from src.pipeline import RBCPipeline

DATASET_FOLDER = "dataset_rbc/images/test"  # Carpeta con las imágenes
ANNOTATIONS_FOLDER = "dataset_rbc/annotations/test"  # Carpeta con los ground truth
OUTPUT_FOLDER = "output"  # Carpeta donde se guardarán los resultados
GRAPHICS_FOLDER = "graphics"
os.makedirs(GRAPHICS_FOLDER, exist_ok=True)

if __name__ == "__main__":
    pipeline = RBCPipeline()
    
    # Obtener lista de imágenes
    image_files = sorted([f for f in os.listdir(DATASET_FOLDER) if f.endswith('.jpg')])
    total_images = len(image_files)
    
    # Almacenar métricas
    all_metrics = {
        'precision': [],
        'recall': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'images_processed': 0,
        'images_failed': 0,
        'failed_images': [],
        'processing_times': [],

        # Solo detecciones reales del modelo
        'y_true_no_fn': [],
        'y_scores_no_fn': [],

        # Detecciones + FN añadidos
        'y_true_with_fn': [],
        'y_scores_with_fn': []
    }
    
    start_time_total = time.time()
    
    print(f"Procesando {total_images} imágenes...")
    print("-" * 80)
    
    for idx, imagen_file in enumerate(image_files, 1):
        nombre_imagen = imagen_file.replace('.jpg', '')
        ruta_imagen = f"{DATASET_FOLDER}/{imagen_file}"
        ruta_xml_gt = f"{ANNOTATIONS_FOLDER}/{nombre_imagen}.xml"
        
        # Verificar que existe el XML correspondiente
        if not os.path.exists(ruta_xml_gt):
            print(f"[{idx}/{total_images}] ⚠️  {nombre_imagen}: XML no encontrado")
            all_metrics['images_failed'] += 1
            all_metrics['failed_images'].append((nombre_imagen, "XML no encontrado"))
            continue
        
        try:
            # Medir tiempo de procesamiento de esta imagen
            start_time_image = time.time()
            
            # Ejecutar pipeline con comparación a ground truth
            metrics = pipeline.procesar_con_gt(ruta_imagen, ruta_xml_gt, OUTPUT_FOLDER)
            
            # Calcular tiempo de procesamiento
            elapsed_time = time.time() - start_time_image
            
            # Acumular valores para curvas ROC, PR y matriz de confusión
            # Caso 1: sin añadir FN artificiales
            y_true_no_fn = list(metrics['y_true'])
            y_scores_no_fn = list(metrics['y_scores'])

            all_metrics['y_true_no_fn'].extend(y_true_no_fn)
            all_metrics['y_scores_no_fn'].extend(y_scores_no_fn)

            # Caso 2: añadiendo FN
            y_true_with_fn = list(metrics['y_true'])
            y_scores_with_fn = list(metrics['y_scores'])

            for _ in range(metrics['FN']):
                y_true_with_fn.append(1)      # RBC real
                y_scores_with_fn.append(0.0)  # no detectado

            all_metrics['y_true_with_fn'].extend(y_true_with_fn)
            all_metrics['y_scores_with_fn'].extend(y_scores_with_fn)

            all_metrics['precision'].append(metrics['precision'])
            all_metrics['recall'].append(metrics['recall'])
            all_metrics['TP'].append(metrics['TP'])
            all_metrics['FP'].append(metrics['FP'])
            all_metrics['FN'].append(metrics['FN'])
            all_metrics['processing_times'].append(elapsed_time)
            all_metrics['images_processed'] += 1
            
            # Mostrar progreso con tiempo
            print(f"[{idx}/{total_images}] ✓ {nombre_imagen}: "
                  f"Prec={metrics['precision']:.2f}, Rec={metrics['recall']:.2f} "
                  f"({elapsed_time:.2f}s)")
            
        except Exception as e:
            print(f"[{idx}/{total_images}] ✗ {nombre_imagen}: Error - {str(e)}")
            all_metrics['images_failed'] += 1
            all_metrics['failed_images'].append((nombre_imagen, str(e)))
    
    total_time = time.time() - start_time_total
    
    # Mostrar resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print(f"Total de imágenes procesadas: {all_metrics['images_processed']}")
    print(f"Total de fallos: {all_metrics['images_failed']}")
    
    # Mostrar tiempos
    if all_metrics['processing_times']:
        avg_time = sum(all_metrics['processing_times']) / len(all_metrics['processing_times'])
        min_time = min(all_metrics['processing_times'])
        max_time = max(all_metrics['processing_times'])
        
        print(f"\nTiempos de procesamiento:")
        print(f"  Tiempo total: {total_time:.2f}s ({total_time/60:.2f} minutos)")
        print(f"  Tiempo promedio por imagen: {avg_time:.2f}s")
        print(f"  Tiempo mínimo: {min_time:.2f}s")
        print(f"  Tiempo máximo: {max_time:.2f}s")
    
    if all_metrics['images_processed'] > 0:
        # Calcular promedios
        avg_precision = sum(all_metrics['precision']) / len(all_metrics['precision'])
        avg_recall = sum(all_metrics['recall']) / len(all_metrics['recall'])
        total_TP = sum(all_metrics['TP'])
        total_FP = sum(all_metrics['FP'])
        total_FN = sum(all_metrics['FN'])
        
        print(f"\nMétricas promedio:")
        print(f"  Precisión: {avg_precision:.4f}")
        print(f"  Recall: {avg_recall:.4f}")
        print(f"  F1-Score: {2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-10):.4f}")
        
        print(f"\nMétricas acumuladas:")
        print(f"  TP (verdaderos positivos): {total_TP}")
        print(f"  FP (falsos positivos): {total_FP}")
        print(f"  FN (falsos negativos): {total_FN}")
        
        tpr = total_TP / (total_TP + total_FN + 1e-10)
        fnr = total_FN / (total_FN + total_TP + 1e-10)
        print(f"\nTasas acumuladas:")
        print(f"  TPR (True Positive Rate): {tpr:.4f}")
        print(f"  FNR (False Negative Rate): {fnr:.4f}")

    # Mostrar imágenes que fallaron
    if all_metrics['failed_images']:
        print(f"\nImágenes con errores ({len(all_metrics['failed_images'])}):")
        for img_name, error in all_metrics['failed_images']:
            print(f"  - {img_name}: {error}")

    
    def save_plots(y_true, y_scores, suffix):
        if len(y_true) == 0 or len(y_scores) == 0:
            print(f"No hay datos para generar gráficas: {suffix}")
            return

        if len(set(y_true)) < 2:
            print(f"No se puede calcular ROC/PR para {suffix}: solo hay una clase.")
            return

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({suffix})")
        plt.legend()
        plt.savefig(os.path.join(GRAPHICS_FOLDER, f"roc_curve_{suffix}.png"))
        plt.close()

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve ({suffix})")
        plt.savefig(os.path.join(GRAPHICS_FOLDER, f"pr_curve_{suffix}.png"))
        plt.close()

        # Confusion Matrix
        y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No RBC", "RBC"]
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, values_format="d", colorbar=False)
        plt.title(f"Confusion Matrix ({suffix})")
        plt.savefig(os.path.join(GRAPHICS_FOLDER, f"confusion_matrix_{suffix}.png"))
        plt.close()

save_plots(
    all_metrics['y_true_no_fn'],
    all_metrics['y_scores_no_fn'],
    "sin_fn"
)

save_plots(
    all_metrics['y_true_with_fn'],
    all_metrics['y_scores_with_fn'],
    "con_fn"
)