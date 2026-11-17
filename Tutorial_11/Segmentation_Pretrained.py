import copy
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import matplotlib.pyplot as plt

DATA_DIR = "/home/umair/Deep_Learning/data/Dataset_Segmentation"
IMG_SIZE = 256
EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-5
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = ["deeplabv3", "maskrcnn"]
PATIENCE = 5             
MIN_EPOCHS = 5           
MIN_DELTA = 1e-4
COMPUTE_TRAIN_MIOU = True 
FLIP_AUGMENT = True     


class DeepLabV3Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        x = (images - self.mean) / self.std
        return self.model(x)["out"]

    def compute_loss(self, batch):
        images = torch.stack([s["image"] for s in batch])
        masks = torch.stack([s["sem_mask"] for s in batch])
        logits = self(images)
        return F.cross_entropy(logits, masks)

    @torch.no_grad()
    def predict(self, batch):
        images = torch.stack([s["image"] for s in batch])
        logits = self(images)
        return logits.argmax(dim=1) 

class MaskRCNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT",
            min_size=IMG_SIZE,
            max_size=IMG_SIZE,
        )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    def compute_loss(self, batch):
        was_training = self.model.training
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        images = [s["image"] for s in batch]
        targets = [
            {"boxes": s["boxes"], "labels": s["labels"], "masks": s["inst_masks"]}
            for s in batch
        ]
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())

        if not was_training:
            self.model.eval()
        return loss

    @torch.no_grad()
    def predict(self, batch, score_thresh=0.5):
        images = [s["image"] for s in batch]
        outputs = self.model(images)

        sem_preds = []
        for out in outputs:
            sem_mask = torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.long, device=DEVICE)
            keep = out["scores"] >= score_thresh
            boxes_scores = out["scores"][keep]
            labels = out["labels"][keep]
            masks = out["masks"][keep, 0] > 0.5

            order = torch.argsort(boxes_scores)
            for idx in order:
                sem_mask[masks[idx]] = labels[idx]
            sem_preds.append(sem_mask)

        return torch.stack(sem_preds)


def calculate_segmentation_metrics(preds, targets, num_classes):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    pixel_acc = float((preds == targets).float().mean().item())

    ious = []
    for c in range(1, num_classes):
        pred_c = preds == c
        target_c = targets == c
        union = (pred_c | target_c).sum().item()
        if union == 0:
            continue
        inter = (pred_c & target_c).sum().item()
        ious.append(inter / union)

    miou = float(np.mean(ious)) if ious else 0.0
    return miou, pixel_acc


def load_dataset(folder):
    raw = []
    for jf in sorted(Path(folder).glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        if len(data.get("shapes", [])) > 0:
            raw.append((jf, data))

    class_names = sorted({s["label"].strip() for _, d in raw for s in d["shapes"]})
    class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}  # 0 = background

    samples = []
    for jf, data in raw:
        img_path = jf.parent / Path(data["imagePath"]).name
        if not img_path.exists():
            matches = [m for m in jf.parent.glob(f"{jf.stem}.*")
                       if m.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif"}]
            if not matches:
                continue
            img_path = matches[0]

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        orig_w, orig_h = img.size

        sem_mask_img = Image.new("L", (orig_w, orig_h), 0)
        sem_draw = ImageDraw.Draw(sem_mask_img)

        boxes, labels, inst_masks = [], [], []
        for s in data["shapes"]:
            label = s["label"].strip()
            cls_id = class_to_idx[label]
            pts = [(float(x), float(y)) for x, y in s["points"]]
            if len(pts) < 3:
                continue

            sem_draw.polygon(pts, fill=cls_id)

            inst_img = Image.new("L", (orig_w, orig_h), 0)
            ImageDraw.Draw(inst_img).polygon(pts, fill=1)
            inst_img = inst_img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            inst_arr = np.array(inst_img, dtype=np.uint8)

            ys, xs = np.where(inst_arr > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)
            inst_masks.append(inst_arr)

        if len(boxes) == 0:
            continue

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        sem_mask_img = sem_mask_img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

        samples.append({
            "image": torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1),
            "sem_mask": torch.from_numpy(np.array(sem_mask_img, dtype=np.int64)),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "inst_masks": torch.tensor(np.array(inst_masks), dtype=torch.uint8),
        })

    return samples, class_names


def prepare_sample(sample, flip=False):
    img = sample["image"].to(DEVICE)
    sem_mask = sample["sem_mask"].to(DEVICE)
    boxes = sample["boxes"].to(DEVICE)
    labels = sample["labels"].to(DEVICE)
    inst_masks = sample["inst_masks"].to(DEVICE)

    if flip and random.random() < 0.5:
        img = img.flip(-1)
        sem_mask = sem_mask.flip(-1)
        inst_masks = inst_masks.flip(-1)
        boxes = boxes.clone()
        boxes[:, [0, 2]] = (IMG_SIZE - 1) - boxes[:, [2, 0]]

    return {
        "image": img,
        "sem_mask": sem_mask,
        "boxes": boxes,
        "labels": labels,
        "inst_masks": inst_masks,
    }

def make_batch(batch, flip=False):
    return [prepare_sample(s, flip=flip) for s in batch]


def train_one_epoch(model, train, optimizer):
    model.train()
    random.shuffle(train)
    total = 0.0
    for i in range(0, len(train), BATCH_SIZE):
        batch = make_batch(train[i:i + BATCH_SIZE], flip=FLIP_AUGMENT)
        if len(batch) < 2:
            continue
        loss = model.compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
    return total / len(train)

@torch.no_grad()
def validation_loss(model, val):
    model.eval()
    total = 0.0
    for i in range(0, len(val), BATCH_SIZE):
        batch = make_batch(val[i:i + BATCH_SIZE])
        total += model.compute_loss(batch).item() * len(batch)
    return total / len(val)


@torch.no_grad()
def evaluate(model, samples, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    for i in range(0, len(samples), BATCH_SIZE):
        batch = make_batch(samples[i:i + BATCH_SIZE])
        preds = model.predict(batch)
        targets = torch.stack([s["sem_mask"] for s in batch])
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    miou, pixel_acc = calculate_segmentation_metrics(all_preds, all_targets, num_classes)
    return miou, pixel_acc, all_preds, all_targets

def train_model(name, train, val, num_classes):
    model = {"deeplabv3": DeepLabV3Model, "maskrcnn": MaskRCNNModel}[name](num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"\n=== {name} | parameters: {sum(p.numel() for p in model.parameters()):,} ===")

    hist = {"train_loss": [], "val_loss": [], "train_miou": [], "val_miou": []}
    best_miou, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train, optimizer)
        val_loss = validation_loss(model, val)
        val_miou, val_acc, _, _ = evaluate(model, val, num_classes)
        train_miou = evaluate(model, train, num_classes)[0] if COMPUTE_TRAIN_MIOU else float("nan")

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_miou"].append(train_miou)
        hist["val_miou"].append(val_miou)
        print(f"Epoch {epoch:2d} | train loss {train_loss:.3f} | val loss {val_loss:.3f} "
              f"| train mIoU {train_miou:.3f} | val mIoU {val_miou:.3f} | val acc {val_acc:.3f}")

        if val_miou > best_miou + MIN_DELTA:
            bad_epochs = 0
        elif epoch > MIN_EPOCHS:
            bad_epochs += 1
        if val_miou >= best_miou:
            best_miou, best_epoch = val_miou, epoch
            best_state = copy.deepcopy(model.state_dict())
        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best val mIoU {best_miou:.3f} "
                  f"at epoch {best_epoch}).")
            break

    del optimizer
    model.load_state_dict(best_state)
    return model, hist, best_epoch

def show_curves(name, hist, best_epoch):
    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, (ax_loss, ax_iou) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, hist["train_loss"], label="Train loss")
    ax_loss.plot(epochs, hist["val_loss"], label="Validation loss")
    ax_loss.set_title(f"{name}: loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    if COMPUTE_TRAIN_MIOU:
        ax_iou.plot(epochs, hist["train_miou"], label="Train mIoU")
    ax_iou.plot(epochs, hist["val_miou"], label="Validation mIoU")
    ax_iou.set_title(f"{name}: mIoU")
    ax_iou.set_xlabel("Epoch")
    ax_iou.set_ylabel("mIoU")

    for ax in (ax_loss, ax_iou):
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
        ax.legend()
    plt.tight_layout()
    plt.show()

def show_predictions(title, test, preds, num_classes, max_samples=4):
    n_show = min(max_samples, len(test))
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    cmap = plt.get_cmap("tab10", num_classes)
    for row in range(n_show):
        img = test[row]["image"]
        true_mask = test[row]["sem_mask"]
        pred_mask = preds[row]

        axes[row, 0].imshow(img.cpu().permute(1, 2, 0).numpy())
        axes[row, 0].set_title("Input Image")

        axes[row, 1].imshow(true_mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[row, 1].set_title("True Mask")

        axes[row, 2].imshow(pred_mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[row, 2].set_title("Predicted Mask")

        for col in range(3):
            axes[row, col].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    samples, class_names = load_dataset(DATA_DIR)
    num_classes = len(class_names) + 1  # foreground classes + background (0)
    print(f"{len(samples)} images, classes ({num_classes} total with bg): {class_names}")

    random.shuffle(samples)
    n_train, n_val = int(0.70 * len(samples)), int(0.15 * len(samples))
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    results = {}
    for name in MODELS:
        model, hist, best_epoch = train_model(name, train, val, num_classes)
        show_curves(name, hist, best_epoch)

        test_miou, test_acc, preds, _ = evaluate(model, test, num_classes)
        results[name] = (test_miou, test_acc)
        print(f"{name} TEST  mIoU: {test_miou:.3f} | pixel acc: {test_acc:.3f}")
        show_predictions(name, test, preds, num_classes)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nModel comparison (test set)")
    print(f"{'model':<12}{'mIoU':>10}{'pixel_acc':>12}")
    for name, (miou, acc) in results.items():
        print(f"{name:<12}{miou:>10.3f}{acc:>12.3f}")

if __name__ == "__main__":
    main()