import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

import matplotlib.pyplot as plt


DATA_DIR = "/home/umair/Deep_Learning/data/Dataset_Segmentation"
IMG_SIZE = 256             
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-5
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = ["unet"]
PATIENCE = 10            
MIN_EPOCHS = 10            
MIN_DELTA = 1e-4
COMPUTE_TRAIN_MIOU = True  
FLIP_AUGMENT = True        


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.enc1 = double_conv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = double_conv(256, 512)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = double_conv(512 + 256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = double_conv(256 + 128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = double_conv(128 + 64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool1(c1))
        c3 = self.enc3(self.pool2(c2))

        b = self.bottleneck(self.pool3(c3))

        u3 = self.dec3(torch.cat([self.up3(b), c3], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), c1], dim=1))
        return self.out_conv(u1)  # (N, C, H, W)

    def compute_loss(self, images, masks):
        logits = self(images)
        return F.cross_entropy(logits, masks)

    @torch.no_grad()
    def predict(self, images):
        logits = self(images)
        return logits.argmax(dim=1)


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
    class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}

    samples = []
    for jf, data in raw:
        img_path = jf.parent / Path(data["imagePath"]).name
        if not img_path.exists():
            matches = [m for m in jf.parent.glob(f"{jf.stem}.*")
                       if m.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif"}]
            if not img_path.exists() and not matches:
                continue
            img_path = matches[0]

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        orig_w, orig_h = img.size

        mask_img = Image.new("L", (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(mask_img)
        for s in data["shapes"]:
            label = s["label"].strip()
            pts = [(float(x), float(y)) for x, y in s["points"]]
            if len(pts) >= 3:
                draw.polygon(pts, fill=class_to_idx[label])

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask_img = mask_img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

        img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.array(mask_img, dtype=np.int64))

        samples.append((img_t.to(DEVICE), mask_t.to(DEVICE)))

    return samples, class_names


def random_flip(img, mask):
    if random.random() < 0.5:
        img = img.flip(-1)
        mask = mask.flip(-1)
    return img, mask


def make_batch(batch, flip=False):
    imgs, masks = [], []
    for img, m in batch:
        if flip:
            img, m = random_flip(img, m)
        imgs.append(img)
        masks.append(m)
    return torch.stack(imgs), torch.stack(masks)


def train_one_epoch(model, train, optimizer):
    model.train()
    random.shuffle(train)
    total = 0.0
    for i in range(0, len(train), BATCH_SIZE):
        batch = train[i:i + BATCH_SIZE]
        images, masks = make_batch(batch, flip=FLIP_AUGMENT)
        loss = model.compute_loss(images, masks)
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
        batch = val[i:i + BATCH_SIZE]
        images, masks = make_batch(batch)
        total += model.compute_loss(images, masks).item() * len(batch)
    return total / len(val)


@torch.no_grad()
def evaluate(model, samples, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]
        images, masks = make_batch(batch)
        preds = model.predict(images)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    miou, pixel_acc = calculate_segmentation_metrics(all_preds, all_targets, num_classes)
    return miou, pixel_acc, all_preds, all_targets


def train_model(name, train, val, num_classes):
    model = {"unet": UNet}[name](num_classes).to(DEVICE)
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
        img, true_mask = test[row]
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

    print("\nModel comparison (test set)")
    print(f"{'model':<12}{'mIoU':>10}{'pixel_acc':>12}")
    for name, (miou, acc) in results.items():
        print(f"{name:<12}{miou:>10.3f}{acc:>12.3f}")


if __name__ == "__main__":
    main()