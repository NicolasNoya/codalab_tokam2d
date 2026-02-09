import sys
from pathlib import Path
import torch
import yaml
import shutil
from ultralytics import YOLO
import numpy as np

# Add ingestion_program to path to import tokam2d_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion_program"))
from tokam2d_utils import TokamDataset


def prepare_yolo_dataset(data_dir, output_dir="yolo_dataset", val_split=0.2):
    """
    Prepare dataset in YOLO format from TokamDataset.

    YOLO format:
    - images/ folder with .jpg or .png files
    - labels/ folder with .txt files (one per image)
    - Each label line: class_id x_center y_center width height (normalized 0-1)

    Args:
        data_dir: Path to TokamDataset directory
        output_dir: Output directory for YOLO dataset
        val_split: Fraction of labeled data for validation

    Returns:
        Path to dataset.yaml file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create directory structure
    train_images = output_path / "images" / "train"
    train_labels = output_path / "labels" / "train"
    val_images = output_path / "images" / "val"
    val_labels = output_path / "labels" / "val"

    for dir_path in [train_images, train_labels, val_images, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = TokamDataset(data_dir, include_unlabeled=True)

    # Separate labeled and unlabeled samples
    labeled_indices = []
    unlabeled_indices = []

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        if (
            target is not None
            and "boxes" in target
            and target["boxes"] is not None
            and len(target["boxes"]) > 0
        ):
            labeled_indices.append(idx)
        else:
            unlabeled_indices.append(idx)

    # Split labeled data
    import random

    random.shuffle(labeled_indices)
    split_point = int((1 - val_split) * len(labeled_indices))
    train_indices = labeled_indices[:split_point]
    val_indices = labeled_indices[split_point:] + unlabeled_indices

    print(f"Preparing YOLO dataset:")
    print(f"  Train samples: {len(train_indices)}")
    print(
        f"  Val samples: {len(val_indices)} ({len(labeled_indices[split_point:])} labeled, {len(unlabeled_indices)} unlabeled)"
    )

    # Save training data
    print("\nSaving training data...")
    for idx in train_indices:
        image, target = dataset[idx]

        # Save image as PNG
        img_np = (image.squeeze().numpy() * 255).astype(np.uint8)
        from PIL import Image

        img_pil = Image.fromarray(img_np, mode="L")
        img_path = train_images / f"frame_{idx:06d}.png"
        img_pil.save(img_path)

        # Save labels in YOLO format
        label_path = train_labels / f"frame_{idx:06d}.txt"
        with open(label_path, "w") as f:
            if (
                target is not None
                and "boxes" in target
                and len(target["boxes"]) > 0
            ):
                boxes = target["boxes"]
                for box in boxes:
                    x, y, w, h = box.numpy()
                    # Convert to YOLO format: class_id x_center y_center width height (normalized)
                    x_center = (x + w / 2) / 512.0
                    y_center = (y + h / 2) / 512.0
                    width = w / 512.0
                    height = h / 512.0
                    # Class 0 = plasma (we have 1 class)
                    f.write(
                        f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )

    # Save validation data
    print("Saving validation data...")
    for idx in val_indices:
        image, target = dataset[idx]

        # Save image
        img_np = (image.squeeze().numpy() * 255).astype(np.uint8)
        from PIL import Image

        img_pil = Image.fromarray(img_np, mode="L")
        img_path = val_images / f"frame_{idx:06d}.png"
        img_pil.save(img_path)

        # Save labels (empty for unlabeled)
        label_path = val_labels / f"frame_{idx:06d}.txt"
        with open(label_path, "w") as f:
            if (
                target is not None
                and "boxes" in target
                and len(target["boxes"]) > 0
            ):
                boxes = target["boxes"]
                for box in boxes:
                    x, y, w, h = box.numpy()
                    x_center = (x + w / 2) / 512.0
                    y_center = (y + h / 2) / 512.0
                    width = w / 512.0
                    height = h / 512.0
                    f.write(
                        f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )

    # Create dataset.yaml
    dataset_yaml = {
        "path": str(output_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "plasma"},
        "nc": 1,  # number of classes
    }

    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"\n✓ YOLO dataset prepared at: {output_path}")
    print(f"✓ Dataset config saved at: {yaml_path}")

    return yaml_path


def train_yolo_model(
    data_yaml, model_size="n", epochs=20, img_size=512, batch_size=16
):
    """
    Train YOLOv8 model on prepared dataset.

    Args:
        data_yaml: Path to dataset.yaml file
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size

    Returns:
        Trained YOLO model
    """
    # Load pretrained YOLO model
    model_name = f"yolov8{model_size}.pt"
    print(f"\nLoading YOLO model: {model_name}")
    model = YOLO(model_name)

    # Train model
    print(f"\n{'='*70}")
    print("TRAINING YOLO MODEL")
    print(f"{'='*70}\n")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="tokam_plasma_detection",
        project="runs/detect",
        # Augmentation settings
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        degrees=15.0,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,  # Scale augmentation
        shear=0.0,  # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.5,  # Vertical flip probability
        fliplr=0.5,  # Horizontal flip probability
        mosaic=1.0,  # Mosaic augmentation probability
        mixup=0.0,  # Mixup augmentation probability
        # Training settings
        optimizer="AdamW",
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Other settings
        patience=50,  # Early stopping patience
        save=True,
        save_period=-1,
        cache=False,
        device=None,  # Auto-select device
        workers=8,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}\n")

    return model


def evaluate_yolo_model(model, data_yaml):
    """
    Evaluate YOLO model on validation set.

    Args:
        model: Trained YOLO model
        data_yaml: Path to dataset.yaml

    Returns:
        Validation results
    """
    print(f"\n{'='*70}")
    print("EVALUATING ON VALIDATION SET")
    print(f"{'='*70}\n")

    results = model.val(data=str(data_yaml))

    # Print metrics
    print(f"\n📊 Validation Metrics:")
    print(f"   mAP@0.5: {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"   Precision: {results.box.mp:.4f}")
    print(f"   Recall: {results.box.mr:.4f}")

    return results


def predict_with_yolo(model, h5_file_path, output_file="yolo_predictions.json"):
    """
    Run YOLO inference on h5 file.

    Args:
        model: Trained YOLO model
        h5_file_path: Path to h5 file
        output_file: Path to save predictions

    Returns:
        dict: Predictions for each frame
    """
    import h5py
    import json

    h5_path = Path(h5_file_path)
    print(f"\n📂 Loading test data from: {h5_path}")

    # Load h5 file
    with h5py.File(h5_path, "r") as f:
        print(f"   Available keys: {list(f.keys())}")

        if "rho" in f:
            data_key = "rho"
        elif "images" in f:
            data_key = "images"
        elif "data" in f:
            data_key = "data"
        else:
            data_key = list(f.keys())[0]
            print(f"   Using key: '{data_key}'")

        images = f[data_key][:]
        print(f"   Loaded {len(images)} frames of shape {images.shape[1:]}")

    # Run predictions
    predictions = {}

    print(f"\n🔍 Running YOLO inference on {len(images)} frames...")

    for frame_idx in range(len(images)):
        # Get image
        img = images[frame_idx]

        # Handle different shapes
        if img.ndim == 3:
            img = img[0]

        # Convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)

        # Run YOLO prediction
        results = model.predict(img_uint8, verbose=False, conf=0.25)

        # Extract boxes
        frame_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Get xyxy format and convert to xywh
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                frame_boxes.append([float(x), float(y), float(w), float(h)])

        predictions[frame_idx] = frame_boxes

        # Progress update
        if (frame_idx + 1) % 50 == 0 or (frame_idx + 1) == len(images):
            num_detections = sum(len(boxes) for boxes in predictions.values())
            print(
                f"   Processed {frame_idx + 1}/{len(images)} frames, Total detections: {num_detections}"
            )

    # Save predictions
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\n✅ Predictions saved to: {output_path}")

    # Summary
    num_frames_with_detections = sum(
        1 for boxes in predictions.values() if len(boxes) > 0
    )
    total_detections = sum(len(boxes) for boxes in predictions.values())

    print(f"\n📊 Inference Summary:")
    print(f"   Total frames: {len(predictions)}")
    print(f"   Frames with detections: {num_frames_with_detections}")
    print(f"   Total detections: {total_detections}")
    print(
        f"   Average detections per frame: {total_detections / len(predictions):.2f}"
    )

    return predictions


if __name__ == "__main__":
    # Example usage
    data_dir = Path("./dev_phase/input_data/train")

    # Prepare dataset
    yaml_path = prepare_yolo_dataset(data_dir)

    # Train model
    model = train_yolo_model(yaml_path, model_size="n", epochs=20)

    # Evaluate
    evaluate_yolo_model(model, yaml_path)

    # Save final model
    model.save("tokam_yolo_final.pt")
    print("\n✓ Model saved as tokam_yolo_final.pt")
