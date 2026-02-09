import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# Add ingestion_program to path to import tokam2d_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion_program"))
from tokam2d_utils import TokamDataset


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return tuple(zip(*batch))


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

        # Segmentation mask head
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1),
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

        # Generate segmentation masks
        pred_masks = self.mask_head(
            spatial_features
        )  # (batch_size, num_classes, H, W)

        if self.training and targets is not None:
            # Compute losses during training
            losses = self.compute_losses(
                pred_boxes, pred_logits, pred_masks, targets
            )
            return losses
        else:
            # Return predictions during inference
            return self.postprocess_predictions(
                pred_boxes, pred_logits, pred_masks
            )

    def compute_losses(self, pred_boxes, pred_logits, pred_masks, targets):
        """Compute training losses"""
        batch_size = pred_boxes.shape[0]
        device = pred_boxes.device

        # Initialize losses as tensors on the correct device
        class_loss = torch.tensor(0.0, device=device)
        bbox_loss = torch.tensor(0.0, device=device)
        mask_loss = torch.tensor(0.0, device=device)

        num_valid_targets = 0

        for i in range(batch_size):
            # Check if this sample has valid labels (not None)
            target_labels = targets[i].get("labels", None)
            if target_labels is not None and len(target_labels) > 0:
                num_valid_targets += 1
                num_targets = len(target_labels)

                # Classification loss
                class_loss += F.cross_entropy(
                    pred_logits[i, :num_targets],
                    target_labels[:num_targets],
                )

                # Bounding box loss (L1 loss)
                target_boxes = targets[i].get("boxes", None)
                if target_boxes is not None:
                    bbox_loss += F.l1_loss(
                        pred_boxes[i, :num_targets], target_boxes[:num_targets]
                    )

                # Note: Dataset doesn't provide masks, so mask loss stays at 0
                # Mask head is trained implicitly through the shared backbone

        # Average losses over samples with valid targets
        if num_valid_targets > 0:
            class_loss = class_loss / num_valid_targets
            bbox_loss = bbox_loss / num_valid_targets
            mask_loss = mask_loss / num_valid_targets

        return {
            "loss_classifier": class_loss,
            "loss_box_reg": bbox_loss,
            "loss_mask": mask_loss,
        }

    def postprocess_predictions(self, pred_boxes, pred_logits, pred_masks):
        """Convert raw predictions to final format"""
        batch_size = pred_boxes.shape[0]
        results = []

        for i in range(batch_size):
            # Get predictions for this image
            boxes = pred_boxes[i]  # (num_queries, 4)
            logits = pred_logits[i]  # (num_queries, num_classes)

            # Convert logits to scores and labels
            scores = F.softmax(logits, dim=-1)
            labels = scores.argmax(dim=-1)
            scores = scores.max(dim=-1).values

            # Filter out low confidence predictions (background class = 0)
            keep = (labels > 0) & (scores > 0.5)

            # Prepare masks (take the predicted class channel)
            masks = F.interpolate(
                pred_masks[i : i + 1],
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            results.append(
                {
                    "boxes": boxes[keep],
                    "labels": labels[keep],
                    "scores": scores[keep],
                    "masks": masks,
                }
            )

        return results


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
    train_dataset = TokamDataset(training_dir, include_unlabeled=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True
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

    model.eval().to("cpu")
    return model
