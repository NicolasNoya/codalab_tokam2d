import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import random
import math

# Add ingestion_program to path to import tokam2d_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion_program"))
from tokam2d_utils import TokamDataset


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return tuple(zip(*batch))


def augment_image_and_boxes(image, boxes, p=0.5):
    """
    Apply random augmentations to image and bounding boxes.

    Args:
        image: (C, H, W) tensor
        boxes: (N, 4) tensor in (x, y, w, h) format, pixel coordinates
        p: probability of applying augmentation

    Returns:
        augmented_image, augmented_boxes
    """
    if boxes is None or len(boxes) == 0:
        return image, boxes

    device = image.device
    dtype = image.dtype
    C, H, W = image.shape

    # Random horizontal flip
    if random.random() < p:
        image = torch.flip(image, dims=[2])  # Flip width
        boxes = boxes.clone()
        boxes[:, 0] = W - (boxes[:, 0] + boxes[:, 2])  # x = W - (x + w)

    # Random vertical flip
    if random.random() < p:
        image = torch.flip(image, dims=[1])  # Flip height
        boxes = boxes.clone()
        boxes[:, 1] = H - (boxes[:, 1] + boxes[:, 3])  # y = H - (y + h)

    # Random rotation (90, 180, 270 degrees)
    if random.random() < p:
        k = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        image = torch.rot90(image, k=k, dims=[1, 2])

        # Rotate boxes
        boxes = boxes.clone()
        for _ in range(k):
            # Rotate 90 degrees clockwise
            x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            # After 90° rotation: (x, y) -> (H - y - h, x)
            new_x = H - y - h
            new_y = x
            new_w = h
            new_h = w
            boxes = torch.stack([new_x, new_y, new_w, new_h], dim=1)
            H, W = W, H  # Swap dimensions

    # Random translation (small shifts)
    if random.random() < p:
        max_shift = min(H, W) * 0.1  # Max 10% shift
        shift_x = random.uniform(-max_shift, max_shift)
        shift_y = random.uniform(-max_shift, max_shift)

        # Shift image (pad and crop)
        pad_x = int(abs(shift_x))
        pad_y = int(abs(shift_y))
        padded = F.pad(
            image.unsqueeze(0), (pad_x, pad_x, pad_y, pad_y), mode="reflect"
        ).squeeze(0)

        # Crop to original size
        start_x = pad_x + int(shift_x)
        start_y = pad_y + int(shift_y)
        image = padded[:, start_y : start_y + H, start_x : start_x + W]

        # Shift boxes
        boxes = boxes.clone()
        boxes[:, 0] += shift_x
        boxes[:, 1] += shift_y

        # Clip boxes to image boundaries
        boxes[:, 0] = torch.clamp(boxes[:, 0], 0, W - 1)
        boxes[:, 1] = torch.clamp(boxes[:, 1], 0, H - 1)
        # Ensure width and height are positive and fit within image
        boxes[:, 2] = torch.clamp(boxes[:, 2], 1, W)
        boxes[:, 3] = torch.clamp(boxes[:, 3], 1, H)
        # Make sure boxes don't extend beyond image boundaries
        boxes[:, 2] = torch.minimum(
            boxes[:, 2],
            torch.tensor(W, dtype=boxes.dtype, device=boxes.device)
            - boxes[:, 0],
        )
        boxes[:, 3] = torch.minimum(
            boxes[:, 3],
            torch.tensor(H, dtype=boxes.dtype, device=boxes.device)
            - boxes[:, 1],
        )

    # Random brightness adjustment
    if random.random() < p:
        brightness_factor = random.uniform(0.8, 1.2)
        image = torch.clamp(image * brightness_factor, 0, 1)

    return image, boxes


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    Boxes are in (x, y, w, h) format.

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)

    Returns:
        iou: (N, M)
    """
    # Convert to (x1, y1, x2, y2)
    boxes1_xyxy = torch.cat(
        [boxes1[:, :2], boxes1[:, :2] + boxes1[:, 2:]], dim=1
    )
    boxes2_xyxy = torch.cat(
        [boxes2[:, :2], boxes2[:, :2] + boxes2[:, 2:]], dim=1
    )

    # Compute intersection
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Compute union
    area1 = boxes1[:, 2] * boxes1[:, 3]  # (N,)
    area2 = boxes2[:, 2] * boxes2[:, 3]  # (M,)
    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou


def generalized_box_iou_loss(pred_boxes, target_boxes):
    """
    Compute Generalized IoU loss.
    GIoU = IoU - |C \ (A ∪ B)| / |C|
    where C is the smallest enclosing box.

    Args:
        pred_boxes: (N, 4) in (x, y, w, h) format, normalized [0, 1]
        target_boxes: (N, 4) in (x, y, w, h) format, normalized [0, 1]

    Returns:
        loss: scalar
    """
    # Convert to (x1, y1, x2, y2)
    pred_xyxy = torch.cat(
        [pred_boxes[:, :2], pred_boxes[:, :2] + pred_boxes[:, 2:]], dim=1
    )
    target_xyxy = torch.cat(
        [target_boxes[:, :2], target_boxes[:, :2] + target_boxes[:, 2:]], dim=1
    )

    # Compute IoU
    lt = torch.max(pred_xyxy[:, :2], target_xyxy[:, :2])
    rb = torch.min(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area_pred = pred_boxes[:, 2] * pred_boxes[:, 3]
    area_target = target_boxes[:, 2] * target_boxes[:, 3]
    union = area_pred + area_target - inter

    iou = inter / (union + 1e-6)

    # Compute enclosing box
    enclose_lt = torch.min(pred_xyxy[:, :2], target_xyxy[:, :2])
    enclose_rb = torch.max(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

    # GIoU
    giou = iou - (enclose_area - union) / (enclose_area + 1e-6)

    # GIoU loss (1 - GIoU)
    loss = 1 - giou
    return loss.mean()


class DINOv3Segmentation(nn.Module):
    """
    DINOv3-based model for instance segmentation of plasma.
    Uses DINOv3 as a backbone with custom heads for detection and segmentation.
    DINOv3 includes register tokens for better dense prediction performance.
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    ):
        super().__init__()

        # Load DINOv3 backbone
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
            self.config = self.backbone.config
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(self.config)

        # Freeze backbone initially (can unfreeze for fine-tuning)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # DINOv3 hidden dimensions (384 for vits, 768 for vitb)
        self.hidden_dim = self.config.hidden_size
        self.num_register_tokens = self.config.num_register_tokens
        self.patch_size = self.config.patch_size

        # Detection head (bounding boxes)
        self.bbox_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # x, y, w, h
        )

        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Object query embeddings for detection (similar to DETR approach)
        self.num_queries = 10  # Maximum number of plasma instances to detect
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, images, targets=None):
        """
        Args:
            images: list of tensors of shape (C, H, W)
            targets: list of dicts with keys 'boxes', 'labels', 'masks' (for training)

        Returns:
            If training: dict of losses
            If inference: list of dicts with 'boxes', 'labels', 'scores', 'masks'
        """
        batch_size = len(images)

        # Prepare batch - ensure all images are same size and have 3 channels
        # DINOv3 expects 3-channel RGB images (224x224)
        processed_images = []
        for img in images:
            # Ensure image has batch dimension
            if img.dim() == 2:  # (H, W)
                img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif img.dim() == 3:  # (C, H, W)
                img = img.unsqueeze(0)  # (1, C, H, W)

            # Resize to 224x224
            img = F.interpolate(
                img,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

            # Convert single channel to 3 channels (grayscale to RGB)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)  # (1, 3, 224, 224)

            processed_images.append(img.squeeze(0))  # (3, 224, 224)

        images_stacked = torch.stack(
            processed_images
        )  # (batch_size, 3, 224, 224)

        # Get DINOv3 features
        # Output shape: (batch_size, num_tokens, hidden_dim)
        # For 224x224 input with patch_size=16: (batch_size, 1 + num_register_tokens + 196, hidden_dim)
        # 1 CLS token + register tokens + 196 patch tokens (14x14)
        outputs = self.backbone(images_stacked)
        features = outputs.last_hidden_state

        # Separate CLS token, register tokens, and patch features
        cls_token = features[:, 0]  # (batch_size, hidden_dim)
        # Skip register tokens for patch features
        patch_features = features[
            :, 1 + self.num_register_tokens :
        ]  # (batch_size, num_patches, hidden_dim)

        # Reshape patch features to spatial grid
        # For 224x224 with patch_size=16: 14x14 patches
        num_patches_per_side = 224 // self.patch_size
        spatial_features = patch_features.transpose(1, 2).reshape(
            batch_size,
            self.hidden_dim,
            num_patches_per_side,
            num_patches_per_side,
        )

        # Query-based detection
        queries = self.query_embed.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        # queries: (batch_size, num_queries, hidden_dim)

        # Use cross-attention between queries and features (simplified version)
        # For simplicity, use average pooling of spatial features
        pooled_features = (
            F.adaptive_avg_pool2d(spatial_features, 1).squeeze(-1).squeeze(-1)
        )

        # Predict boxes and classes for each query
        pred_boxes = self.bbox_head(queries)  # (batch_size, num_queries, 4)
        pred_logits = self.class_head(
            queries
        )  # (batch_size, num_queries, num_classes)

        if self.training and targets is not None:
            # Compute losses during training
            losses = self.compute_losses(pred_boxes, pred_logits, targets)
            return losses
        else:
            # Return predictions during inference
            return self.postprocess_predictions(pred_boxes, pred_logits)

    def compute_losses(self, pred_boxes, pred_logits, targets):
        """Compute training losses"""
        batch_size = pred_boxes.shape[0]
        device = pred_boxes.device

        # Initialize losses as tensors on the correct device
        class_loss = torch.tensor(0.0, device=device)
        bbox_loss = torch.tensor(0.0, device=device)

        num_valid_targets = 0

        for i in range(batch_size):
            # Check if this sample has valid labels (not None)
            target_labels = targets[i].get("labels", None)
            if target_labels is not None and len(target_labels) > 0:
                num_valid_targets += 1
                num_targets = min(len(target_labels), self.num_queries)

                # Classification loss
                class_loss += F.cross_entropy(
                    pred_logits[i, :num_targets],
                    target_labels[:num_targets],
                )

                # Bounding box loss (GIoU loss)
                target_boxes = targets[i].get("boxes", None)
                if target_boxes is not None:
                    # Normalize ground truth boxes to [0, 1] range
                    # Images are 512x512 in original coordinates
                    normalized_target_boxes = target_boxes[:num_targets].clone()
                    normalized_target_boxes[
                        :, [0, 2]
                    ] /= 512.0  # Normalize x, w
                    normalized_target_boxes[
                        :, [1, 3]
                    ] /= 512.0  # Normalize y, h

                    # Use GIoU loss instead of L1
                    bbox_loss += generalized_box_iou_loss(
                        pred_boxes[i, :num_targets], normalized_target_boxes
                    )

        # Average losses over samples with valid targets
        if num_valid_targets > 0:
            class_loss = class_loss / num_valid_targets
            bbox_loss = bbox_loss / num_valid_targets

        return {
            "loss_classifier": class_loss,
            "loss_box_reg": bbox_loss,
        }

    def postprocess_predictions(self, pred_boxes, pred_logits):
        """Convert raw predictions to final format"""
        batch_size = pred_boxes.shape[0]
        results = []

        for i in range(batch_size):
            # Get predictions for this image
            boxes = pred_boxes[
                i
            ]  # (num_queries, 4) in normalized coordinates [0, 1]
            logits = pred_logits[i]  # (num_queries, num_classes)

            # Keep boxes in normalized [0, 1] range
            # Multiply by 512 to convert to pixel coordinates if needed
            normalized_boxes = boxes.clone()

            # Convert logits to scores and labels
            scores = F.softmax(logits, dim=-1)
            labels = scores.argmax(dim=-1)
            scores = scores.max(dim=-1).values

            # Filter out low confidence predictions (background class = 0)
            keep = (labels > 0) & (scores > 0.5)

            results.append(
                {
                    "boxes": normalized_boxes[keep],
                    "labels": labels[keep],
                    "scores": scores[keep],
                }
            )

        return results


def split_dataset(dataset):
    """
    Split dataset into train and validation sets.
    Samples without bounding boxes go to validation set.
    Samples with bounding boxes are split 80/20 for train/val.

    Returns:
        train_indices, val_indices
    """
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

    # Split labeled data 80/20
    import random

    random.shuffle(labeled_indices)
    split_point = int(0.8 * len(labeled_indices))
    train_indices = labeled_indices[:split_point]
    val_labeled_indices = labeled_indices[split_point:]

    # All unlabeled go to validation
    val_indices = val_labeled_indices + unlabeled_indices

    return train_indices, val_indices


def train_model(
    training_dir, model_name="facebook/dinov3-vits16-pretrain-lvd1689m"
):
    """
    Train DINOv3-based segmentation model for plasma detection.

    Args:
        training_dir: Path to training data directory
        model_name: DINOv3 model to use (default: vits16, can use vitb16 or vit7b16 for larger models)

    Returns:
        Trained DINOv3Segmentation model
    """
    # Load full dataset
    full_dataset = TokamDataset(training_dir, include_unlabeled=True)

    # Split into train and validation
    train_indices, val_indices = split_dataset(full_dataset)

    print(f"Dataset split:")
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")

    # Create train and validation subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize DINOv3 segmentation model
    print(f"Loading DINOv3 model: {model_name}")
    model = DINOv3Segmentation(
        num_classes=2, pretrained=True, model_name=model_name
    )
    model.to(device)
    model.train()

    # Only train the heads initially, keep backbone frozen
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataloader)
    )

    max_epochs = 1

    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        epoch_loss = 0

        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = [im.to(device) for im in images]
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            optimizer.zero_grad()
            loss_dict = model(images, targets)

            # Combine all losses
            total_loss = sum(loss for loss in loss_dict.values())
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {total_loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

    # Return model and validation dataloader for evaluation
    model.eval().to("cpu")
    return model, val_dataloader, full_dataset


def visualize_validation_results(
    model, val_dataloader, dataset, device="cpu", num_samples=6
):
    """
    Visualize model predictions on validation set.

    Args:
        model: Trained model
        val_dataloader: Validation dataloader
        dataset: Original dataset to access images
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    model.to(device)
    model.eval()

    # Collect samples
    samples_collected = 0
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    with torch.no_grad():
        for images, targets in val_dataloader:
            for i in range(len(images)):
                if samples_collected >= num_samples:
                    break

                # Get image
                img = images[i]
                target = targets[i]

                # Prepare for inference
                if img.dim() == 2:
                    img_input = img.unsqueeze(0)
                elif img.dim() == 3 and img.shape[0] == 1:
                    img_input = img
                else:
                    img_input = img

                # Convert to 3 channels if needed
                if img_input.shape[0] == 1:
                    img_input = img_input.repeat(3, 1, 1)

                # Get predictions
                predictions = model([img_input.to(device)])
                pred = predictions[0]

                # Visualize
                ax = axes[samples_collected]

                # Show image
                img_viz = img.squeeze().cpu().numpy()
                ax.imshow(img_viz, cmap="viridis")

                # Draw ground truth boxes (if any)
                has_gt = False
                if (
                    target is not None
                    and "boxes" in target
                    and target["boxes"] is not None
                    and len(target["boxes"]) > 0
                ):
                    has_gt = True
                    for box in target["boxes"]:
                        x, y, w, h = box.cpu().numpy()
                        rect = patches.Rectangle(
                            (x, y),
                            w,
                            h,
                            linewidth=2,
                            edgecolor="lime",
                            facecolor="none",
                            label="Ground Truth",
                        )
                        ax.add_patch(rect)

                # Draw predictions
                if len(pred["boxes"]) > 0:
                    for box, score in zip(pred["boxes"], pred["scores"]):
                        # Convert normalized to pixel coordinates
                        x, y, w, h = box.cpu().numpy()
                        x_pix, y_pix, w_pix, h_pix = (
                            x * 512,
                            y * 512,
                            w * 512,
                            h * 512,
                        )

                        rect = patches.Rectangle(
                            (x_pix, y_pix),
                            w_pix,
                            h_pix,
                            linewidth=2,
                            edgecolor="red",
                            facecolor="none",
                            linestyle="--",
                            label="Prediction",
                        )
                        ax.add_patch(rect)

                        # Add score
                        ax.text(
                            x_pix,
                            y_pix - 5,
                            f"{score.item():.2f}",
                            color="red",
                            fontsize=9,
                            bbox=dict(
                                boxstyle="round", facecolor="white", alpha=0.7
                            ),
                        )

                # Title
                title = f"{'Labeled' if has_gt else 'Unlabeled'} - {len(pred['boxes'])} detections"
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.axis("off")

                samples_collected += 1

            if samples_collected >= num_samples:
                break

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="lime", lw=2, label="Ground Truth"),
        Line2D([0], [0], color="red", lw=2, linestyle="--", label="Prediction"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=2,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.98),
    )

    plt.suptitle(
        "Validation Set Results", fontsize=16, fontweight="bold", y=0.95
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print(f"\n✓ Visualized {samples_collected} validation samples")
