import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssd import SSDClassificationHead

import matplotlib.pyplot as plt

DATA_DIR = "/home/umair/Deep_Learning/data/Dataset_Detection"
IMG_SIZE = 300             
EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-4
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRETRAINED = True         
PATIENCE = 5             
MIN_EPOCHS = 5           
MIN_DELTA = 1e-4
COMPUTE_TRAIN_MAP = True

def make_model(num_classes):
    weights = SSD300_VGG16_Weights.DEFAULT if PRETRAINED else None
    model = ssd300_vgg16(weights=weights, weights_backbone=None)     # 91 COCO classes

    in_channels = det_utils.retrieve_out_channels(model.backbone, (IMG_SIZE, IMG_SIZE))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
    return model.to(DEVICE)


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    iw = max(0, min(x2, a2) - max(x1, a1))
    ih = max(0, min(y2, b2) - max(y1, b1))
    inter = iw * ih
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - inter
    return inter / union if union > 0 else 0


def calculate_precision_recall(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels,
                               iou_threshold=0.5, score_threshold=0.5):
    tp, fp, matched = 0, 0, set()
    for i, pb in enumerate(pred_boxes):
        if pred_scores[i] < score_threshold:
            continue
        best_iou, best_j = 0, -1
        for j, gb in enumerate(gt_boxes):
            if pred_labels[i] == gt_labels[j] and j not in matched:
                iou = calculate_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
        if best_iou >= iou_threshold:
            tp += 1
            matched.add(best_j)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall


def calculate_ap(precisions, recalls):
    precisions = [0.0] + list(precisions) + [0.0]
    recalls = [0.0] + list(recalls) + [1.0]
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    idx = [i for i in range(1, len(recalls)) if recalls[i] != recalls[i - 1]]
    return sum((recalls[i] - recalls[i - 1]) * precisions[i] for i in idx)


def calculate_map(predictions, ground_truths, num_classes, iou_threshold=0.5):
    aps = []
    for c in range(1, num_classes):
        detections, n_gt = [], 0
        for img, (p, g) in enumerate(zip(predictions, ground_truths)):
            n_gt += sum(1 for l in g["labels"] if l == c)
            for b, s, l in zip(p["boxes"], p["scores"], p["labels"]):
                if l == c:
                    detections.append((s, img, b))
        if n_gt == 0:
            continue
        detections.sort(key=lambda d: d[0], reverse=True)
        matched, tp, fp, precisions, recalls = set(), 0, 0, [], []
        for score, img, box in detections:
            g = ground_truths[img]
            best_iou, best_j = 0, -1
            for j, (gb, gl) in enumerate(zip(g["boxes"], g["labels"])):
                if gl == c and (img, j) not in matched:
                    iou = calculate_iou(box, gb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
            if best_iou >= iou_threshold:
                tp += 1
                matched.add((img, best_j))
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / n_gt)
        aps.append(calculate_ap(precisions, recalls))
    return float(np.mean(aps)) if aps else 0.0


def load_dataset(folder):
    raw = []
    for jf in sorted(Path(folder).glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        if len(data["shapes"]) > 0:
            raw.append((jf, data))

    class_names = sorted({s["label"].strip() for _, d in raw for s in d["shapes"]})
    class_to_idx = {n: i + 1 for i, n in enumerate(class_names)}     # 0 = background

    samples = []
    for jf, data in raw:
        img = Image.open(jf.parent / Path(data["imagePath"]).name)
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        boxes, labels = [], []
        for s in data["shapes"]:
            pts = np.array(s["points"], dtype=np.float32)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            x1, x2 = x1 * IMG_SIZE / w, x2 * IMG_SIZE / w
            y1, y2 = y1 * IMG_SIZE / h, y2 * IMG_SIZE / h
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(class_to_idx[s["label"].strip()])
        if len(boxes) == 0:
            continue
        samples.append((img.to(DEVICE),
                        torch.tensor(boxes, dtype=torch.float32, device=DEVICE),
                        torch.tensor(labels, dtype=torch.long, device=DEVICE)))
    return samples, class_names


def batch_loss(model, batch):
    images = [b[0] for b in batch]
    targets = [{"boxes": b[1], "labels": b[2]} for b in batch]
    return sum(model(images, targets).values())          # box regression + classification


def train_one_epoch(model, train, optimizer):
    model.train()
    random.shuffle(train)
    total = 0
    for i in range(0, len(train), BATCH_SIZE):
        batch = train[i:i + BATCH_SIZE]
        loss = batch_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
    return total / len(train)


@torch.no_grad()
def validation_loss(model, val):
    model.train()
    total = 0
    for i in range(0, len(val), BATCH_SIZE):
        batch = val[i:i + BATCH_SIZE]
        total += batch_loss(model, batch).item() * len(batch)
    return total / len(val)


@torch.no_grad()
def evaluate(model, samples, num_classes):
    model.eval()
    predictions, ground_truths = [], []
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]
        outs = model([b[0] for b in batch])
        for out, (_, boxes, labels) in zip(outs, batch):
            predictions.append({"boxes": out["boxes"].tolist(),
                                "scores": out["scores"].tolist(),
                                "labels": out["labels"].tolist()})
            ground_truths.append({"boxes": boxes.tolist(), "labels": labels.tolist()})
    return calculate_map(predictions, ground_truths, num_classes), predictions, ground_truths


def train_model(train, val, num_classes):
    model = make_model(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"SSD300-VGG16 | parameters: {sum(p.numel() for p in model.parameters()):,}")

    hist = {"train_loss": [], "val_loss": [], "train_map": [], "val_map": []}
    best_map, best_state, best_epoch, bad_epochs = -1, None, 0, 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train, optimizer)
        val_loss = validation_loss(model, val)
        val_map = evaluate(model, val, num_classes)[0]
        train_map = evaluate(model, train, num_classes)[0] if COMPUTE_TRAIN_MAP else float("nan")

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_map"].append(train_map)
        hist["val_map"].append(val_map)
        print(f"Epoch {epoch:2d} | train loss {train_loss:.3f} | val loss {val_loss:.3f} "
              f"| train mAP {train_map:.3f} | val mAP {val_map:.3f}")

        if val_map > best_map + MIN_DELTA:
            bad_epochs = 0
        elif epoch > MIN_EPOCHS:
            bad_epochs += 1
        if val_map >= best_map:                         
            best_map, best_epoch = val_map, epoch
            best_state = copy.deepcopy(model.state_dict())
        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best val mAP {best_map:.3f} "
                  f"at epoch {best_epoch}).")
            break

    model.load_state_dict(best_state)
    return model, hist, best_epoch


def show_curves(hist, best_epoch):
    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, (ax_loss, ax_map) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, hist["train_loss"], label="Train loss")
    ax_loss.plot(epochs, hist["val_loss"], label="Validation loss")
    ax_loss.set_title("SSD: loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    if COMPUTE_TRAIN_MAP:
        ax_map.plot(epochs, hist["train_map"], label="Train mAP@0.5")
    ax_map.plot(epochs, hist["val_map"], label="Validation mAP@0.5")
    ax_map.set_title("SSD: mAP")
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("mAP@0.5")

    for ax in (ax_loss, ax_map):
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
        ax.legend()
    plt.tight_layout()
    plt.show()


def show_predictions(test, predictions, class_names, score_thresh=0.5):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (img, boxes, _), p in zip(axes.flat, test, predictions):
        ax.imshow(img.cpu().permute(1, 2, 0).numpy())
        for b in boxes.tolist():                                      
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, color="lime", linewidth=2))
        for b, s, l in zip(p["boxes"], p["scores"], p["labels"]):    
            if s >= score_thresh:
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                           fill=False, color="red", linewidth=2))
                ax.text(b[0], b[1], f"{class_names[l - 1]} {s:.2f}", color="white",
                        fontsize=8, backgroundcolor="red")
    plt.suptitle("SSD (pretrained)  (green = ground truth, red = prediction)")
    plt.show()


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    samples, class_names = load_dataset(DATA_DIR)
    num_classes = len(class_names) + 1
    print(f"{len(samples)} images, classes: {class_names}")

    random.shuffle(samples)
    n_train, n_val = int(0.70 * len(samples)), int(0.15 * len(samples))
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    model, hist, best_epoch = train_model(train, val, num_classes)
    show_curves(hist, best_epoch)

    test_map, predictions, ground_truths = evaluate(model, test, num_classes)
    pr = [calculate_precision_recall(p["boxes"], p["scores"], p["labels"], g["boxes"], g["labels"])
          for p, g in zip(predictions, ground_truths)]
    print(f"\nTEST  mAP@0.5: {test_map:.3f} | precision: {np.mean([x[0] for x in pr]):.3f} "
          f"| recall: {np.mean([x[1] for x in pr]):.3f}")
    show_predictions(test, predictions, class_names)


if __name__ == "__main__":
    main()
