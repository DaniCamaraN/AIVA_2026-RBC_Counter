import random
import shutil
from pathlib import Path
from typing import List, Tuple

class DatasetSplitter:

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        annotations_dir: str,
        output_base: str = "dataset_rbc",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> None:
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Los ratios train, val y test deben sumar 1.")

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.annotations_dir = Path(annotations_dir)

        self.output_base = Path(output_base)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Imágenes
        self.train_images_dir = self.output_base / "images" / "train"
        self.val_images_dir = self.output_base / "images" / "val"
        self.test_images_dir = self.output_base / "images" / "test"

        # Labels YOLO
        self.train_labels_dir = self.output_base / "labels" / "train"
        self.val_labels_dir = self.output_base / "labels" / "val"
        self.test_labels_dir = self.output_base / "labels" / "test"

        # XML ground truth
        self.train_annotations_dir = self.output_base / "annotations" / "train"
        self.val_annotations_dir = self.output_base / "annotations" / "val"
        self.test_annotations_dir = self.output_base / "annotations" / "test"

        self.valid_exts = {".jpg", ".jpeg", ".png"}

    def create_dirs(self) -> None:
        for d in [
            self.train_images_dir,
            self.val_images_dir,
            self.test_images_dir,
            self.train_labels_dir,
            self.val_labels_dir,
            self.test_labels_dir,
            self.train_annotations_dir,
            self.val_annotations_dir,
            self.test_annotations_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def collect_pairs(self) -> List[Tuple[Path, Path, Path]]:
        pairs = []

        for img_path in self.images_dir.iterdir():
            if img_path.suffix.lower() not in self.valid_exts:
                continue

            label_path = self.labels_dir / f"{img_path.stem}.txt"
            annotation_path = self.annotations_dir / f"{img_path.stem}.xml"

            if not label_path.exists():
                print(f"Label no encontrado para: {img_path.name}")
                continue

            if not annotation_path.exists():
                print(f"XML no encontrado para: {img_path.name}")
                continue

            pairs.append((img_path, label_path, annotation_path))

        print(f"Total pares válidos: {len(pairs)}")
        return pairs

    def split(self, pairs):
        random.seed(self.seed)
        random.shuffle(pairs)

        total = len(pairs)

        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        test_pairs = pairs[val_end:]

        print(f"Train: {len(train_pairs)}")
        print(f"Val: {len(val_pairs)}")
        print(f"Test: {len(test_pairs)}")

        return train_pairs, val_pairs, test_pairs

    def copy_pairs(self, pairs_list, dst_img_dir, dst_lbl_dir, dst_ann_dir):
        for img_path, label_path, annotation_path in pairs_list:
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            shutil.copy2(label_path, dst_lbl_dir / label_path.name)
            shutil.copy2(annotation_path, dst_ann_dir / annotation_path.name)

    def create_yaml(self) -> None:
        yaml_path = self.output_base / "rbc.yaml"

        content = (
            f"path: {self.output_base}\n"
            "train: images/train\n"
            "val: images/val\n"
            "\n"
            "names:\n"
            "  0: RBC\n"
        )

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"YAML generado en: {yaml_path}")

    def run(self) -> None:
        print("Preparando dataset YOLO...")

        self.create_dirs()
        pairs = self.collect_pairs()

        train_pairs, val_pairs, test_pairs = self.split(pairs)

        self.copy_pairs(
            train_pairs,
            self.train_images_dir,
            self.train_labels_dir,
            self.train_annotations_dir
        )

        self.copy_pairs(
            val_pairs,
            self.val_images_dir,
            self.val_labels_dir,
            self.val_annotations_dir
        )

        self.copy_pairs(
            test_pairs,
            self.test_images_dir,
            self.test_labels_dir,
            self.test_annotations_dir
        )

        self.create_yaml()

        print("Dataset dividido correctamente.")
