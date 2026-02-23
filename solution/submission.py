import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import torch

from tokam2d_utils import TokamDataset


class YOLOWrapper(torch.nn.Module):
    """
    Wrapper to make YOLO model compatible with ingestion program format.

    The ingestion program expects:
    - Input: tuple of images from collate_fn
    - Output: list of dicts with "boxes" and "scores" keys
    """

    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model

    def forward(self, images):
        """
        Args:
            images: tuple of tensors (from collate_fn)

        Returns:
            list of dicts with "boxes" (Tensor[N, 4]) and "scores" (Tensor[N])
        """
        results = []

        for img in images:
            # Convert tensor to numpy for YOLO
            # img is shape (1, H, W) - grayscale
            img_np = (
                img.squeeze().cpu().numpy()
                if img.is_cuda
                else img.squeeze().numpy()
            )

            # Normalize to 0-255
            img_norm = (img_np - img_np.min()) / (
                img_np.max() - img_np.min() + 1e-8
            )
            img_uint8 = (img_norm * 255).astype(np.uint8)

            # Convert grayscale to RGB (YOLO expects 3 channels)
            # Stack the grayscale image 3 times to create RGB
            img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

            # Run YOLO prediction
            pred = self.yolo_model(img_rgb, verbose=False)

            # Extract boxes and scores
            if len(pred) > 0 and len(pred[0].boxes) > 0:
                boxes = pred[0].boxes.xyxy.cpu()  # [N, 4] in xyxy format
                scores = pred[0].boxes.conf.cpu()  # [N]
            else:
                # No detections
                boxes = torch.zeros((0, 4))
                scores = torch.zeros((0,))

            results.append({"boxes": boxes, "scores": scores})

        return results

    def eval(self):
        """Set to evaluation mode."""
        self.yolo_model.model.eval()
        return self

    def to(self, device):
        """Move to device."""
        # YOLO handles device internally
        return self


def train_model(training_dir):
    """
    Train a YOLO model for plasma blob detection.

    This function loads the pre-trained YOLO model and fine-tunes it
    on the tokamak dataset. The training configuration matches the
    parameters from train_yolo_model.ipynb.

    Args:
        training_dir: Directory containing training data

    Returns:
        Trained YOLO model
    """
    import torch
    import shutil
    import yaml

    # Configuration matching train_yolo_model.ipynb
    CONFIG = {
        "model": "yolov10l.pt",
        "epochs": 10,
        "batch_size": 4,
        "img_size": 1024,
        "patience": 300,
        "val_split": 0,
        "val": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "workers": 0,
        "optimizer": "Adam",
        "lr0": 1e-4,
        "weight_decay": 0.0005,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 20,
        "translate": 0.05,
        "scale": 1.0,
        "flipud": 0,
        "fliplr": 0,
        "mosaic": 0.5,
    }

    # Setup directories using absolute paths to avoid issues in Docker
    import os

    cwd = Path(os.getcwd())
    YOLO_DATASET_DIR = cwd / "yolo_training_data"
    OUTPUT_DIR = cwd / "runs"

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Current working directory: {cwd}")
    print(f"Dataset directory: {YOLO_DATASET_DIR}")

    print("Loading dataset...")
    dataset = TokamDataset(training_dir, include_unlabeled=False)

    # Find labeled samples
    labeled_indices = []
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        if target["boxes"] is not None and len(target["boxes"]) > 0:
            labeled_indices.append(idx)

    print(f"Total labeled samples: {len(labeled_indices)}")

    # Use ALL data for training (no validation split)
    np.random.seed(42)
    np.random.shuffle(labeled_indices)
    train_indices = labeled_indices  # Use all data for training
    # Create minimal validation set (just 1 sample) to satisfy YOLO requirements
    val_indices = [labeled_indices[0]]

    print(f"Training: {len(train_indices)} samples (using all labeled data)")
    print(f"Validation: {len(val_indices)} samples (minimal set for YOLO)")

    # Prepare YOLO format dataset
    def save_yolo_format(dataset, indices, split_name, output_dir):
        img_dir = output_dir / "images" / split_name
        label_dir = output_dir / "labels" / split_name

        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(indices):
            image, target = dataset[idx]
            img_np = (
                image.squeeze().cpu().numpy()
                if hasattr(image, "cpu")
                else image.squeeze().numpy()
            )

            # Normalize to 0-255
            img_norm = (img_np - img_np.min()) / (
                img_np.max() - img_np.min() + 1e-8
            )
            img_uint8 = (img_norm * 255).astype(np.uint8)

            # Convert grayscale to RGB (YOLO expects 3 channels)
            img_rgb = Image.fromarray(img_uint8).convert("RGB")

            # Save image
            img_name = f"{split_name}_{idx:06d}.jpg"
            img_rgb.save(img_dir / img_name)

            # Save labels
            label_name = f"{split_name}_{idx:06d}.txt"
            label_path = label_dir / label_name

            with open(label_path, "w") as f:
                if target["boxes"] is not None:
                    for box in target["boxes"]:
                        x, y, w, h = (
                            box.cpu().numpy()
                            if hasattr(box, "cpu")
                            else box.numpy()
                        )

                        img_h, img_w = img_np.shape
                        if x > 1 or y > 1:
                            x, y, w, h = (
                                x / img_w,
                                y / img_h,
                                w / img_w,
                                h / img_h,
                            )

                        # Convert corner to center format
                        x_center = x + w / 2
                        y_center = y + h / 2

                        # Clamp to [0, 1]
                        x_center = np.clip(x_center, 0, 1)
                        y_center = np.clip(y_center, 0, 1)
                        w = np.clip(w, 0, 1)
                        h = np.clip(h, 0, 1)

                        f.write(
                            f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                        )

    # Clean and create dataset
    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)

    save_yolo_format(dataset, train_indices, "train", YOLO_DATASET_DIR)
    save_yolo_format(dataset, val_indices, "val", YOLO_DATASET_DIR)

    # Verify dataset was created
    train_img_count = len(
        list((YOLO_DATASET_DIR / "images" / "train").glob("*.jpg"))
    )
    val_img_count = len(
        list((YOLO_DATASET_DIR / "images" / "val").glob("*.jpg"))
    )
    print(f"\nDataset verification:")
    print(f"  Train images: {train_img_count}")
    print(f"  Val images: {val_img_count} (minimal set)")

    if train_img_count == 0:
        raise RuntimeError(
            f"Dataset creation failed! Train images: {train_img_count}"
        )

    # Create YAML configuration
    config_yaml = {
        "path": str(YOLO_DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["plasma"],
    }

    yaml_path = YOLO_DATASET_DIR / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_yaml, f)

    print(f"Dataset preparation complete!")
    print(f"Config file: {yaml_path}")
    print(f"Dataset path in config: {config_yaml['path']}")

    # Train model
    print(f"\nInitializing {CONFIG['model']}...")
    model = YOLO(CONFIG["model"])

    print("\nStarting training...")
    results = model.train(
        data=str(yaml_path),
        epochs=CONFIG["epochs"],
        batch=CONFIG["batch_size"],
        imgsz=CONFIG["img_size"],
        device=CONFIG["device"],
        workers=CONFIG["workers"],
        patience=CONFIG["patience"],
        save_period=10,
        project=str(OUTPUT_DIR),
        name="detect",
        exist_ok=True,
        pretrained=True,
        optimizer=CONFIG["optimizer"],
        lr0=CONFIG["lr0"],
        weight_decay=CONFIG["weight_decay"],
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        val=CONFIG["val"],
        hsv_h=CONFIG["hsv_h"],
        hsv_s=CONFIG["hsv_s"],
        hsv_v=CONFIG["hsv_v"],
        degrees=CONFIG["degrees"],
        translate=CONFIG["translate"],
        scale=CONFIG["scale"],
        flipud=CONFIG["flipud"],
        fliplr=CONFIG["fliplr"],
        mosaic=CONFIG["mosaic"],
    )

    print("\nTraining complete!")

    # Get the actual save directory from training results
    # YOLO saves results and the trainer object contains the save directory
    save_dir = None
    if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir"):
        save_dir = Path(model.trainer.save_dir)
        print(f"\nTraining saved to: {save_dir}")

    # Try to find best.pt using the actual save directory first
    if save_dir and (save_dir / "weights" / "best.pt").exists():
        best_model_path = save_dir / "weights" / "best.pt"
        print(f"Found best model at: {best_model_path}")
    else:
        # Fallback to searching for the file
        import glob

        print("\nSearching for best.pt...")
        search_patterns = [
            str(OUTPUT_DIR / "**" / "best.pt"),
            "runs/**/best.pt",
        ]

        best_model_path = None
        for pattern in search_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                best_model_path = Path(matches[0])
                print(f"Found best model at: {best_model_path}")
                break

        if best_model_path is None:
            raise FileNotFoundError(
                f"Could not find best.pt. Training save directory was: {save_dir}\n"
                f"Please check the directory structure in: {OUTPUT_DIR}"
            )

    yolo_model = YOLO(str(best_model_path))

    # Wrap YOLO model to match ingestion program's expected interface
    wrapped_model = YOLOWrapper(yolo_model)
    wrapped_model.eval()

    print("\nModel wrapped and ready for evaluation")
    return wrapped_model
