import sys
import os
import yaml
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

# Add ingestion_program to path
sys.path.insert(0, 'ingestion_program')
from tokam2d_utils import TokamDataset
def initiliaze():
    # Install ultralytics if needed
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO is installed")
    except ImportError:
        print("Installing ultralytics...")

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Paths
    DATA_DIR = Path('./dev_phase/input_data/train')
    YOLO_DATASET_DIR = Path('./yolo_training_data')
    OUTPUT_DIR = Path('./runs')

    # Training parameters
    CONFIG = {
        # Model
        'model': 'yolov10l.pt',  # nano (fastest), or yolov8s/m/l/x for better accuracy
        
        # Training
        'epochs': 500,
        'batch_size': 4,
        'img_size': 1024,
        'patience': 300,  # Early stopping
        
        # Data split
        'val_split': 0.05,  # 10% for validation
        
        # Hardware
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        
        # Optimization
        'optimizer': 'Adam',
        'lr0': 1e-4,
        'weight_decay': 0.0005,
        
        # Augmentation
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'degrees': 20,  # rotation
        'translate': 0.05,
        'scale': 1.0,
        'flipud': 0,
        'fliplr': 0,
        'mosaic': 0.5,
    }

def save_yolo_format(dataset, indices, split_name, output_dir):
    """
    Save dataset in YOLO format.
    """
    img_dir = output_dir / 'images' / split_name
    label_dir = output_dir / 'labels' / split_name
    
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {split_name} set ({len(indices)} samples)...")
    
    for i, idx in enumerate(tqdm(indices, desc=f"{split_name}")):
        image, target = dataset[idx]
        img_np = image.squeeze().cpu().numpy() if hasattr(image, 'cpu') else image.squeeze().numpy()
        
        # Normalize to 0-255
        img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        
        # Save image
        img_name = f"{split_name}_{idx:06d}.jpg"
        Image.fromarray(img_uint8).save(img_dir / img_name)
        
        # Save labels
        label_name = f"{split_name}_{idx:06d}.txt"
        label_path = label_dir / label_name
        
        with open(label_path, 'w') as f:
            if target['boxes'] is not None:
                for box in target['boxes']:
                    x, y, w, h = box.cpu().numpy() if hasattr(box, 'cpu') else box.numpy()
                    
                    # Check if already normalized or in pixels
                    img_h, img_w = img_np.shape
                    if x > 1 or y > 1:
                        x, y, w, h = x/img_w, y/img_h, w/img_w, h/img_h
                    
                    # Convert corner to center format
                    x_center = x + w/2
                    y_center = y + h/2
                    
                    # Clamp to [0, 1]
                    x_center = np.clip(x_center, 0, 1)
                    y_center = np.clip(y_center, 0, 1)
                    w = np.clip(w, 0, 1)
                    h = np.clip(h, 0, 1)
                    
                    # Write: class x_center y_center width height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# Create dataset
save_yolo_format(dataset, train_indices, 'train', YOLO_DATASET_DIR)
save_yolo_format(dataset, val_indices, 'val', YOLO_DATASET_DIR)

print("\n✓ Dataset preparation complete!")

# Initialize model
print(f"Loading {CONFIG['model']}...")
model = YOLO(CONFIG['model'])

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Device: {CONFIG['device']}")
print(f"Epochs: {CONFIG['epochs']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Image size: {CONFIG['img_size']}")
print("="*60 + "\n")

# Train
results = model.train(
    data=str(yaml_path),
    epochs=CONFIG['epochs'],
    batch=CONFIG['batch_size'],
    imgsz=CONFIG['img_size'],
    device=CONFIG['device'],
    workers=CONFIG['workers'],
    patience=CONFIG['patience'],
    save_period=10,
    project=str(OUTPUT_DIR),
    name='detect',
    exist_ok=True,
    pretrained=True,
    optimizer=CONFIG['optimizer'],
    lr0=CONFIG['lr0'],
    weight_decay=CONFIG['weight_decay'],
    verbose=True,
    seed=42,
    deterministic=True,
    single_cls=True,
    cos_lr=True,
    close_mosaic=10,
    amp=True,
    # Augmentation
    hsv_h=CONFIG['hsv_h'],
    hsv_s=CONFIG['hsv_s'],
    hsv_v=CONFIG['hsv_v'],
    degrees=CONFIG['degrees'],
    translate=CONFIG['translate'],
    scale=CONFIG['scale'],
    flipud=CONFIG['flipud'],
    fliplr=CONFIG['fliplr'],
    mosaic=CONFIG['mosaic'],
)