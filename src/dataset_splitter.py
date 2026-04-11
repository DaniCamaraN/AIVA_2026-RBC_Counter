import random
import shutil
from pathlib import Path
from typing import List, Tuple

class DatasetSplitter:

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        output_base: str = "dataset_rbc",
        train_ratio: float = 0.9,
        seed: int = 42
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_base = Path(output_base)
        self.train_ratio = train_ratio
        self.seed = seed

        # Rutas de salida
        self.train_images_dir = self.output_base / "images" / "train"
        self.val_images_dir = self.output_base / "images" / "test"
        self.train_labels_dir = self.output_base / "labels" / "train"
        self.val_labels_dir = self.output_base / "labels" / "test"

        self.valid_exts = {".jpg", ".jpeg", ".png"}

    def create_dirs(self) -> None:
        for d in [
            self.train_images_dir,
            self.val_images_dir,
            self.train_labels_dir,
            self.val_labels_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def collect_pairs(self) -> List[Tuple[Path, Path]]:
        pairs = []

        for img_path in self.images_dir.iterdir():
            if img_path.suffix.lower() not in self.valid_exts:
                continue

            label_path = self.labels_dir / f"{img_path.stem}.txt"

            if label_path.exists():
                pairs.append((img_path, label_path))
            else:
                print(f"Label no encontrado para: {img_path.name}")

        print(f"Total pares válidos: {len(pairs)}")
        return pairs

    def split(self, pairs: List[Tuple[Path, Path]]):
        random.seed(self.seed)
        random.shuffle(pairs)

        split_idx = int(len(pairs) * self.train_ratio)

        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        print(f"Train: {len(train_pairs)}")
        print(f"Test: {len(val_pairs)}")

        return train_pairs, val_pairs

    def copy_pairs(self, pairs_list, dst_img_dir, dst_lbl_dir):
        for img_path, label_path in pairs_list:
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            shutil.copy2(label_path, dst_lbl_dir / label_path.name)

    def create_yaml(self) -> None:
        yaml_path = self.output_base / "rbc.yaml"

        content = (
            f"path: {self.output_base}\n"
            "train: images/train\n"
            "val: images/test\n"
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

        train_pairs, val_pairs = self.split(pairs)

        self.copy_pairs(train_pairs, self.train_images_dir, self.train_labels_dir)
        self.copy_pairs(val_pairs, self.val_images_dir, self.val_labels_dir)

        self.create_yaml()

        print("Dataset dividido correctamente.")