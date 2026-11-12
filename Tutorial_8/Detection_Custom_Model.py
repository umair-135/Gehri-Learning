import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.ops import RoIPool, batched_nms, box_iou

import matplotlib.pyplot as plt


DATA_DIR = "/home/umair/Deep_Learning/data/Dataset_Detection"
IMG_SIZE = 224
EPOCHS = 30
LR = 1e-4
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PATIENCE = 5
MIN_DELTA = 1e-4          
COMPUTE_TRAIN_MAP = True


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(512, 512, 3, padding=1)
        self.cls = nn.Conv2d(512, 9 * 2, 1)    # object / not-object for 9 anchors
        self.bbox = nn.Conv2d(512, 9 * 4, 1)   # 4 box values for 9 anchors

    def forward(self, feature_map):
        x = F.relu(self.conv(feature_map))
        return self.cls(x), self.bbox(x)



class ROIPooling(nn.Module):
    def __init__(self, pool_size=(7, 7)):
        super().__init__()
        self.pool = RoIPool(output_size=pool_size, spatial_scale=1 / 8)  # 1/8 = backbone stride

    def forward(self, feature_map, rois):
        return self.pool(feature_map, rois)



class Heads(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dense = nn.Linear(512 * 7 * 7, 1024)
        self.classifier = nn.Linear(1024, num_classes)
        self.bbox_regressor = nn.Linear(1024, num_classes * 4)

    def forward(self, pooled_rois):
        x = F.relu(self.dense(pooled_rois.flatten(1)))
        return self.classifier(x), self.bbox_regressor(x)   # raw scores (no softmax)



class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.rpn = RPN()
        self.roi_pooling = ROIPooling((7, 7))
        self.heads = Heads(num_classes)

    def forward(self, images, rois):
        feature_map = self.backbone(images)
        rpn_class, rpn_bbox = self.rpn(feature_map)
        pooled_rois = self.roi_pooling(feature_map, rois)
        classifier, bbox_regressor = self.heads(pooled_rois)
        return {"rpn_class": rpn_class, "rpn_bbox": rpn_bbox,
                "classifier": classifier, "bbox_regressor": bbox_regressor}



def calculate_iou(box1, box2):
    a1, b1, a2, b2 = box1
    c1, d1, c2, d2 = box2
    inter_min_a, inter_min_b = max(a1, c1), max(b1, d1)
    inter_max_a, inter_max_b = min(a2, c2), min(b2, d2)
    inter_area = max(0, inter_max_a - inter_min_a) * max(0, inter_max_b - inter_min_b)
    area1 = (a2 - a1) * (b2 - b1)
    area2 = (c2 - c1) * (d2 - d1)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0



def calculate_precision_recall(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels,
                               iou_threshold=0.5, score_threshold=0.5):
    tp, fp = 0, 0
    matched_gt = set()
    for i, pred_box in enumerate(pred_boxes):
        if pred_scores[i] < score_threshold:
            continue
        best_iou, best_gt_index = 0, -1
        for j, gt_box in enumerate(gt_boxes):
            if pred_labels[i] == gt_labels[j] and j not in matched_gt:
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_index = iou, j
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_index)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall



def calculate_ap(precisions, recalls):
    precisions = [0.0] + list(precisions) + [0.0]
    recalls = [0.0] + list(recalls) + [1.0]
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    indices = [i for i in range(1, len(recalls)) if recalls[i] != recalls[i - 1]]
    return sum((recalls[i] - recalls[i - 1]) * precisions[i] for i in indices)


def calculate_map(predictions, ground_truths, num_classes, iou_threshold=0.5):
    average_precisions = []
    for class_id in range(1, num_classes):             # class 0 is background
        detections, n_gt = [], 0
        for img, (p, g) in enumerate(zip(predictions, ground_truths)):
            n_gt += sum(1 for l in g["labels"] if l == class_id)
            for b, s, l in zip(p["boxes"], p["scores"], p["labels"]):
                if l == class_id:
                    detections.append((s, img, b))
        if n_gt == 0:                                   # class not in this data -> skip
            continue

        detections.sort(key=lambda d: d[0], reverse=True)   # best score first
        matched, tp, fp = set(), 0, 0
        precisions, recalls = [], []
        for score, img, box in detections:
            g = ground_truths[img]
            best_iou, best_j = 0, -1
            for j, (gb, gl) in enumerate(zip(g["boxes"], g["labels"])):
                if gl == class_id and (img, j) not in matched:
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
        average_precisions.append(calculate_ap(precisions, recalls))
    return float(np.mean(average_precisions)) if average_precisions else 0.0



def load_dataset(folder):
    raw = []
    for jf in sorted(Path(folder).glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        if len(data["shapes"]) > 0:
            raw.append((jf, data))

    class_names = sorted({s["label"].strip() for _, d in raw for s in d["shapes"]})
    class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}   # 0 = background

    samples = []
    for jf, data in raw:
        img = Image.open(jf.parent / Path(data["imagePath"]).name)
        img = ImageOps.exif_transpose(img).convert("RGB")   # fixes rotated phone photos
        w, h = img.size
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        boxes, labels = [], []
        for s in data["shapes"]:
            pts = np.array(s["points"], dtype=np.float32)
            x1, y1 = pts.min(axis=0)                         # corners can be in any order
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



def encode(roi, gt):
    rw, rh = roi[:, 2] - roi[:, 0], roi[:, 3] - roi[:, 1]
    rx, ry = roi[:, 0] + rw / 2, roi[:, 1] + rh / 2
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gx, gy = gt[:, 0] + gw / 2, gt[:, 1] + gh / 2
    return torch.stack([(gx - rx) / rw, (gy - ry) / rh, torch.log(gw / rw), torch.log(gh / rh)], dim=1)


def decode(roi, d):
    rw, rh = roi[:, 2] - roi[:, 0], roi[:, 3] - roi[:, 1]
    rx, ry = roi[:, 0] + rw / 2, roi[:, 1] + rh / 2
    cx, cy = d[:, 0] * rw + rx, d[:, 1] * rh + ry
    w = torch.exp(d[:, 2].clamp(max=4)) * rw
    h = torch.exp(d[:, 3].clamp(max=4)) * rh
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)



def make_training_rois(gt_boxes, gt_labels, n_jitter=64, n_random=64):
    dev = gt_boxes.device

    base = gt_boxes[torch.randint(len(gt_boxes), (n_jitter,), device=dev)]
    wh = (base[:, 2:] - base[:, :2]).repeat(1, 2)                    # (w, h, w, h)
    jittered = base + (torch.rand(n_jitter, 4, device=dev) - 0.5) * 0.5 * wh

    w = 30 + torch.rand(n_random, device=dev) * 150
    h = 30 + torch.rand(n_random, device=dev) * 150
    x1 = torch.rand(n_random, device=dev) * (IMG_SIZE - w)
    y1 = torch.rand(n_random, device=dev) * (IMG_SIZE - h)
    random_boxes = torch.stack([x1, y1, x1 + w, y1 + h], dim=1)

    rois = torch.cat([gt_boxes, jittered, random_boxes]).clamp(0, IMG_SIZE)
    keep = ((rois[:, 2] - rois[:, 0]) >= 8) & ((rois[:, 3] - rois[:, 1]) >= 8)
    rois = rois[keep]

    max_iou, matched = box_iou(rois, gt_boxes).max(dim=1)
    labels = gt_labels[matched].clone()
    labels[max_iou < 0.5] = 0                                         # background
    targets = encode(rois, gt_boxes[matched])
    return rois, labels, targets



def make_candidate_boxes(sizes=(64, 112, 160, 208), step=24):
    boxes = []
    for w in sizes:
        for h in sizes:
            for x in range(0, IMG_SIZE - w + 1, step):
                for y in range(0, IMG_SIZE - h + 1, step):
                    boxes.append([x, y, x + w, y + h])
    return torch.tensor(boxes, dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def predict(model, img, candidates, num_classes, score_thresh=0.05):
    model.eval()
    out = model(img[None], [candidates])
    probs = F.softmax(out["classifier"], dim=1)                       # (R, num_classes)
    deltas = out["bbox_regressor"].view(-1, num_classes, 4)

    boxes, scores, labels = [], [], []
    for c in range(1, num_classes):
        boxes.append(decode(candidates, deltas[:, c]).clamp(0, IMG_SIZE))
        scores.append(probs[:, c])
        labels.append(torch.full((len(candidates),), c, device=DEVICE))
    boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)

    keep = scores > score_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = batched_nms(boxes, scores, labels, 0.3)[:20]               # remove duplicate boxes
    return boxes[keep].tolist(), scores[keep].tolist(), labels[keep].tolist()


def evaluate(model, samples, candidates, num_classes):
    predictions, ground_truths = [], []
    for img, boxes, labels in samples:
        b, s, l = predict(model, img, candidates, num_classes)
        predictions.append({"boxes": b, "scores": s, "labels": l})
        ground_truths.append({"boxes": boxes.tolist(), "labels": labels.tolist()})
    return calculate_map(predictions, ground_truths, num_classes), predictions, ground_truths



def compute_loss(model, img, rois, roi_labels, targets, num_classes):
    out = model(img[None], [rois])
    cls_loss = F.cross_entropy(out["classifier"], roi_labels)
    deltas = out["bbox_regressor"].view(-1, num_classes, 4)
    pos = roi_labels > 0
    reg_loss = F.smooth_l1_loss(deltas[pos, roi_labels[pos]], targets[pos]) if pos.any() else 0
    return cls_loss + reg_loss


@torch.no_grad()
def validation_loss(model, val, val_rois, num_classes):
    model.eval()
    total = 0
    for (img, _, _), (rois, roi_labels, targets) in zip(val, val_rois):
        total += float(compute_loss(model, img, rois, roi_labels, targets, num_classes))
    return total / len(val)


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

    model = FasterRCNN(num_classes).to(DEVICE)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    candidates = make_candidate_boxes()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    val_rois = [make_training_rois(b, l) for _, b, l in val]

    train_loss_history, val_loss_history = [], []
    train_map_history, val_map_history = [], []
    best_map, best_state, best_epoch = -1, None, 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(train)
        total = 0
        for img, boxes, labels in train:
            rois, roi_labels, targets = make_training_rois(boxes, labels)
            loss = compute_loss(model, img, rois, roi_labels, targets, num_classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)

        val_map, _, _ = evaluate(model, val, candidates, num_classes)
        val_loss = validation_loss(model, val, val_rois, num_classes)
        train_map = evaluate(model, train, candidates, num_classes)[0] if COMPUTE_TRAIN_MAP else float("nan")

        train_loss_history.append(total / len(train))
        val_loss_history.append(val_loss)
        train_map_history.append(train_map)
        val_map_history.append(val_map)
        print(f"Epoch {epoch:2d} | train loss {train_loss_history[-1]:.3f} | val loss {val_loss:.3f} "
              f"| train mAP {train_map:.3f} | val mAP {val_map:.3f}")

        if val_map > best_map + MIN_DELTA:
            best_map, best_epoch = val_map, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}: no val mAP improvement for "
                      f"{PATIENCE} epochs (best {best_map:.3f} at epoch {best_epoch}).")
                break


    model.load_state_dict(best_state)
    test_map, predictions, ground_truths = evaluate(model, test, candidates, num_classes)

    precisions, recalls = [], []
    for p, g in zip(predictions, ground_truths):
        pr, rc = calculate_precision_recall(p["boxes"], p["scores"], p["labels"],
                                            g["boxes"], g["labels"])
        precisions.append(pr)
        recalls.append(rc)
    print(f"\nTEST  mAP@0.5: {test_map:.3f} | precision: {np.mean(precisions):.3f} "
          f"| recall: {np.mean(recalls):.3f}")


    epochs_run = range(1, len(train_loss_history) + 1)
    fig, (ax_loss, ax_map) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs_run, train_loss_history, label="Train loss")
    ax_loss.plot(epochs_run, val_loss_history, label="Validation loss")
    ax_loss.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    if COMPUTE_TRAIN_MAP:
        ax_map.plot(epochs_run, train_map_history, label="Train mAP@0.5")
    ax_map.plot(epochs_run, val_map_history, label="Validation mAP@0.5")
    ax_map.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("mAP@0.5")
    ax_map.set_title("mAP")
    ax_map.legend()

    plt.tight_layout()
    plt.show()

    # Test predictions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (img, boxes, labels), p in zip(axes.flat, test, predictions):
        ax.imshow(img.cpu().permute(1, 2, 0).numpy())
        for b in boxes.tolist():                                     # real boxes = green
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, color="lime", linewidth=2))
        for b, s, l in zip(p["boxes"], p["scores"], p["labels"]):    # predictions = red
            if s >= 0.3:
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                           fill=False, color="red", linewidth=2))
                ax.text(b[0], b[1], f"{class_names[l - 1]} {s:.2f}", color="white",
                        fontsize=8, backgroundcolor="red")
        ax.axis("off")
    plt.suptitle("Green = ground truth, Red = prediction")
    plt.show()


if __name__ == "__main__":
    main()