import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from tokam2d_utils import TokamDataset


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
        "epochs": 500,
        "batch_size": 4,
        "img_size": 1024,
        "patience": 300,
        "val_split": 0.05,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "workers": 4,
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

    # Setup directories
    YOLO_DATASET_DIR = Path("./yolo_training_data")
    OUTPUT_DIR = Path("./runs")

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = TokamDataset(training_dir, include_unlabeled=False)

    # Find labeled samples
    labeled_indices = []
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        if target["boxes"] is not None and len(target["boxes"]) > 0:
            labeled_indices.append(idx)

    print(f"Total labeled samples: {len(labeled_indices)}")

    # Split dataset
    np.random.seed(42)
    np.random.shuffle(labeled_indices)
    val_size = int(len(labeled_indices) * CONFIG["val_split"])
    train_indices = labeled_indices[val_size:]
    val_indices = labeled_indices[:val_size]

    print(f"Training: {len(train_indices)}, Validation: {len(val_indices)}")

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

            # Save image
            img_name = f"{split_name}_{idx:06d}.jpg"
            Image.fromarray(img_uint8).save(img_dir / img_name)

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

    # Create YAML configuration
    config_yaml = {
        "path": str(YOLO_DATASET_DIR.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["plasma"],
    }

    yaml_path = YOLO_DATASET_DIR / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_yaml, f)

    print("Dataset preparation complete!")

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

    # Load best model - check multiple possible paths
    possible_paths = [
        OUTPUT_DIR
        / "detect"
        / "runs"
        / "detect"
        / "weights"
        / "best.pt",  # Nested path that YOLO creates
        OUTPUT_DIR / "detect" / "weights" / "best.pt",
        OUTPUT_DIR / "detect" / "train" / "weights" / "best.pt",
        Path("runs")
        / "detect"
        / "runs"
        / "detect"
        / "weights"
        / "best.pt",  # Nested path
        Path("runs") / "detect" / "weights" / "best.pt",
        Path("runs") / "detect" / "train" / "weights" / "best.pt",
    ]

    best_model_path = None
    for path in possible_paths:
        if path.exists():
            best_model_path = path
            print(f"\nFound best model at: {best_model_path}")
            break

    if best_model_path is None:
        # Try to find it by searching
        import glob

        search_patterns = [
            str(OUTPUT_DIR / "**" / "best.pt"),
            "runs/**/best.pt",
        ]
        for pattern in search_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                best_model_path = Path(matches[0])
                print(f"\nFound best model at: {best_model_path}")
                break

    if best_model_path is None:
        raise FileNotFoundError(
            f"Could not find best.pt in expected locations. "
            f"Checked: {[str(p) for p in possible_paths]}"
        )

    yolo_model = YOLO(str(best_model_path))
    model = yolo_model.model.to("cpu").eval()
    return model
