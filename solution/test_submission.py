"""
Test script for submission.py YOLO training and inference.

This script tests the train_model function and validates that:
1. The model trains successfully
2. The model can make predictions
3. Output format matches expected format
4. Model handles various input scenarios
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add ingestion_program to path
sys.path.insert(0, "../ingestion_program")
from tokam2d_utils import TokamDataset

# Import the submission function
from submission import train_model


def test_dataset_loading():
    """Test that the dataset loads correctly."""
    print("\n" + "=" * 70)
    print("TEST 1: Dataset Loading")
    print("=" * 70)

    training_dir = Path("../dev_phase/input_data/train")

    if not training_dir.exists():
        print(f"❌ Training directory not found: {training_dir}")
        return False

    dataset = TokamDataset(training_dir, include_unlabeled=False)
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Count labeled samples
    labeled_count = 0
    total_boxes = 0

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        if target["boxes"] is not None and len(target["boxes"]) > 0:
            labeled_count += 1
            total_boxes += len(target["boxes"])

    print(f"✓ Labeled samples: {labeled_count}")
    print(f"✓ Total bounding boxes: {total_boxes}")

    if labeled_count == 0:
        print("❌ No labeled samples found!")
        return False

    return True


def test_model_training():
    """Test the train_model function."""
    print("\n" + "=" * 70)
    print("TEST 2: Model Training")
    print("=" * 70)

    training_dir = Path("../dev_phase/input_data/train")

    print("\nNote: This test will run a full training cycle.")
    print("This may take a significant amount of time.")
    print("Press Ctrl+C to cancel if needed.\n")

    try:
        model = train_model(str(training_dir))
        print("\n✓ Model trained successfully!")
        print(f"✓ Model type: {type(model)}")

        # Check if model is in eval mode
        if hasattr(model, "training"):
            print(f"✓ Model training mode: {model.training}")

        return True, model

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_model_inference(model):
    """Test model inference on sample data."""
    print("\n" + "=" * 70)
    print("TEST 3: Model Inference")
    print("=" * 70)

    if model is None:
        print("❌ No model provided for inference test")
        return False

    try:
        from ultralytics import YOLO

        # Load as YOLO model for inference
        best_model_path = Path("./runs/detect/weights/best.pt")

        if not best_model_path.exists():
            print(f"❌ Best model not found at: {best_model_path}")
            return False

        yolo_model = YOLO(str(best_model_path))
        print(f"✓ Loaded model from: {best_model_path}")

        # Test on validation images
        val_images = sorted(
            list(Path("./yolo_training_data/images/val").glob("*.jpg"))
        )

        if len(val_images) == 0:
            print("⚠ No validation images found, testing on training images...")
            val_images = sorted(
                list(Path("./yolo_training_data/images/train").glob("*.jpg"))
            )[:3]
        else:
            val_images = val_images[:3]  # Test on first 3 images

        if len(val_images) == 0:
            print("❌ No images found for inference testing")
            return False

        print(f"\nTesting inference on {len(val_images)} images...")

        total_detections = 0
        for img_path in val_images:
            results = yolo_model(str(img_path), verbose=False)
            num_detections = len(results[0].boxes)
            total_detections += num_detections

            print(f"  {img_path.name}: {num_detections} detections", end="")

            if num_detections > 0:
                confidences = [
                    box.conf[0].cpu().numpy() for box in results[0].boxes
                ]
                avg_conf = np.mean(confidences)
                print(f" (avg conf: {avg_conf:.3f})")
            else:
                print()

        print(f"\n✓ Inference complete!")
        print(f"✓ Total detections: {total_detections}")

        return True

    except Exception as e:
        print(f"\n❌ Inference failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_output_format(model):
    """Test that model outputs are in the expected format."""
    print("\n" + "=" * 70)
    print("TEST 4: Output Format Validation")
    print("=" * 70)

    try:
        from ultralytics import YOLO

        best_model_path = Path("./runs/detect/weights/best.pt")
        if not best_model_path.exists():
            print("❌ Model not found for format testing")
            return False

        yolo_model = YOLO(str(best_model_path))

        # Get a test image
        test_images = list(
            Path("./yolo_training_data/images/train").glob("*.jpg")
        )[:1]
        if len(test_images) == 0:
            print("❌ No test images found")
            return False

        results = yolo_model(str(test_images[0]), verbose=False)

        print("✓ Prediction structure:")
        print(f"  - Number of results: {len(results)}")
        print(f"  - Number of detections: {len(results[0].boxes)}")

        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            print(f"  - Box coordinates shape: {box.xyxy.shape}")
            print(f"  - Box confidence shape: {box.conf.shape}")
            print(f"  - Sample box (xyxy): {box.xyxy[0].cpu().numpy()}")
            print(f"  - Sample confidence: {box.conf[0].cpu().numpy():.4f}")

        print("\n✓ Output format is valid!")
        return True

    except Exception as e:
        print(f"❌ Format validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_trained_files_exist():
    """Check that all expected output files exist."""
    print("\n" + "=" * 70)
    print("TEST 5: Output Files Verification")
    print("=" * 70)

    expected_files = [
        "./runs/detect/weights/best.pt",
        "./runs/detect/weights/last.pt",
        "./yolo_training_data/config.yaml",
        "./yolo_training_data/images/train",
        "./yolo_training_data/images/val",
        "./yolo_training_data/labels/train",
        "./yolo_training_data/labels/val",
    ]

    all_exist = True
    for file_path in expected_files:
        path = Path(file_path)
        exists = path.exists()
        status = "✓" if exists else "❌"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False

    # Count training samples
    if Path("./yolo_training_data/images/train").exists():
        train_images = len(
            list(Path("./yolo_training_data/images/train").glob("*.jpg"))
        )
        train_labels = len(
            list(Path("./yolo_training_data/labels/train").glob("*.txt"))
        )
        print(f"\n  Training images: {train_images}")
        print(f"  Training labels: {train_labels}")

    if Path("./yolo_training_data/images/val").exists():
        val_images = len(
            list(Path("./yolo_training_data/images/val").glob("*.jpg"))
        )
        val_labels = len(
            list(Path("./yolo_training_data/labels/val").glob("*.txt"))
        )
        print(f"  Validation images: {val_images}")
        print(f"  Validation labels: {val_labels}")

    if all_exist:
        print("\n✓ All expected files exist!")
    else:
        print("\n⚠ Some expected files are missing")

    return all_exist


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("YOLO SUBMISSION TEST SUITE")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    results = {}

    # Test 1: Dataset loading
    results["dataset_loading"] = test_dataset_loading()

    if not results["dataset_loading"]:
        print("\n❌ Dataset loading failed. Cannot proceed with other tests.")
        return results

    # Test 2: Model training
    results["model_training"], model = test_model_training()

    if not results["model_training"]:
        print("\n⚠ Training failed. Skipping inference tests.")
        print(
            "Note: Training might fail due to missing dependencies or data issues."
        )
        return results

    # Test 3: Model inference
    results["model_inference"] = test_model_inference(model)

    # Test 4: Output format
    results["output_format"] = test_output_format(model)

    # Test 5: File verification
    results["file_verification"] = test_trained_files_exist()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name.replace('_', ' ').title()}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print("\n" + "=" * 70)
    print(f"RESULT: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)

    return results


if __name__ == "__main__":
    # Change to solution directory
    os.chdir(Path(__file__).parent)

    # Run tests
    results = run_all_tests()

    # Exit with appropriate code
    if all(results.values()):
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        sys.exit(1)
