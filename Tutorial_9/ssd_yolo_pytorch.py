import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.ops import batched_nms, box_iou

import matplotlib.pyplot as plt


DATA_DIR = "/home/umair/Deep_Learning/data/Dataset_Detection"
IMG_SIZE = 256             # divisible by 64 (SSD use stride 16, 32, 64)
EPOCHS = 60
BATCH_SIZE = 8
LR = 1e-3
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = ["ssd", "yolo"]
PATIENCE = 10              # early stopping
MIN_EPOCHS = 10            # patience not counted before this
MIN_DELTA = 1e-4
COMPUTE_TRAIN_MAP = True   # train mAP each epoch
FLIP_AUGMENT = True        # random horizontal flip while training


def conv_block(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_c), nn.LeakyReLU(0.1))



def postprocess(boxes, class_scores, score_thresh=0.05, nms_iou=0.45, top_k=20):
    """boxes: (N, 4) pixel xyxy. class_scores: (N, C) for the C real classes (ids 1..C)."""
    n, c = class_scores.shape
    labels = torch.arange(1, c + 1, device=boxes.device).repeat(n)
    scores = class_scores.reshape(-1)
    boxes = boxes.unsqueeze(1).expand(n, c, 4).reshape(-1, 4)

    keep = scores > score_thresh
    boxes, scores, labels = boxes[keep].clamp(0, IMG_SIZE), scores[keep], labels[keep]
    keep = batched_nms(boxes, scores, labels, nms_iou)[:top_k]
    return boxes[keep].tolist(), scores[keep].tolist(), labels[keep].tolist()


class SSD(nn.Module):
    SCALES = [0.15, 0.4, 0.7, 0.95]      # anchor size as a fraction of the image
    RATIOS = [1.0, 2.0, 0.5]             # aspect ratios (w / h)

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes   # includes background (class 0)
        self.backbone = nn.Sequential(
            conv_block(3, 32), nn.MaxPool2d(2),
            conv_block(32, 64), nn.MaxPool2d(2),
            conv_block(64, 128), nn.MaxPool2d(2),
            conv_block(128, 256), nn.MaxPool2d(2),          # stride 16
        )
        self.extra = nn.ModuleList([                         # down-sample
            nn.Sequential(conv_block(256, 256), nn.MaxPool2d(2)),
            nn.Sequential(conv_block(256, 256), nn.MaxPool2d(2)),
        ])
        a = 2 * len(self.RATIOS)                             # anchors per cell
        self.cls_heads = nn.ModuleList([nn.Conv2d(256, a * num_classes, 3, padding=1) for _ in range(3)])
        self.box_heads = nn.ModuleList([nn.Conv2d(256, a * 4, 3, padding=1) for _ in range(3)])
        self.register_buffer("anchors", self.make_anchors([IMG_SIZE // 16, IMG_SIZE // 32, IMG_SIZE // 64]))

    def make_anchors(self, sizes):
        anchors = []
        for k, fs in enumerate(sizes):
            s, s_next = self.SCALES[k], self.SCALES[k + 1]
            shapes = [(size * r ** 0.5, size / r ** 0.5)     # (w, h)
                      for size in (s, (s * s_next) ** 0.5) for r in self.RATIOS]
            for y in range(fs):
                for x in range(fs):
                    for w, h in shapes:
                        anchors.append([(x + 0.5) / fs, (y + 0.5) / fs, w, h])   # cx, cy, w, h
        return torch.tensor(anchors, dtype=torch.float32)

    def forward(self, x):
        f = self.backbone(x)
        feats = [f]
        for e in self.extra:
            f = e(f)
            feats.append(f)
        n = x.size(0)
        cls = [h(f).permute(0, 2, 3, 1).reshape(n, -1, self.num_classes)
               for h, f in zip(self.cls_heads, feats)]
        box = [h(f).permute(0, 2, 3, 1).reshape(n, -1, 4) for h, f in zip(self.box_heads, feats)]
        return torch.cat(cls, 1), torch.cat(box, 1)          # (n, N, C), (n, N, 4)

    def encode(self, gt_xyxy):                               # gt box offsets from its anchor
        a = self.anchors
        gcx, gcy = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2, (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2
        gw = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(min=1e-6)
        gh = (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(min=1e-6)
        return torch.stack([(gcx - a[:, 0]) / (a[:, 2] * 0.1), (gcy - a[:, 1]) / (a[:, 3] * 0.1),
                            torch.log(gw / a[:, 2]) / 0.2, torch.log(gh / a[:, 3]) / 0.2], dim=1)

    def decode(self, d):                                     # offsets normalized xyxy
        a = self.anchors
        cx, cy = a[:, 0] + d[:, 0] * 0.1 * a[:, 2], a[:, 1] + d[:, 1] * 0.1 * a[:, 3]
        w = a[:, 2] * torch.exp((d[:, 2] * 0.2).clamp(max=4))
        h = a[:, 3] * torch.exp((d[:, 3] * 0.2).clamp(max=4))
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

    def compute_loss(self, images, boxes_list, labels_list):
        cls_out, box_out = self(images)
        n, num_anchors, _ = cls_out.shape
        a = self.anchors
        anchors_xyxy = torch.stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2,
                                    a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], dim=1)

        cls_t = torch.zeros(n, num_anchors, dtype=torch.long, device=images.device)
        box_t = torch.zeros(n, num_anchors, 4, device=images.device)
        for i, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
            gt = boxes / IMG_SIZE
            iou = box_iou(gt, anchors_xyxy)                  # (num_gt, num_anchors)
            best_iou, best_gt = iou.max(dim=0)               # best gt for every anchor
            best_anchor = iou.argmax(dim=1)                  # best anchor for every gt
            best_iou[best_anchor] = 2.0                      # every gt gets at least one anchor
            best_gt[best_anchor] = torch.arange(len(gt), device=gt.device)
            pos = best_iou >= 0.5
            cls_t[i] = torch.where(pos, labels[best_gt], torch.zeros_like(best_gt))
            box_t[i] = self.encode(gt[best_gt])

        pos = cls_t > 0
        n_pos = pos.sum().clamp(min=1)
        box_loss = F.smooth_l1_loss(box_out[pos], box_t[pos], reduction="sum")

        ce = F.cross_entropy(cls_out.reshape(-1, self.num_classes), cls_t.reshape(-1),
                             reduction="none").view(n, num_anchors)
        neg_ce = ce.detach().clone()                        
        neg_ce[pos] = 0
        rank = neg_ce.argsort(dim=1, descending=True).argsort(dim=1)
        neg = rank < 3 * pos.sum(dim=1, keepdim=True)
        cls_loss = ce[pos | neg].sum()
        return (box_loss + cls_loss) / n_pos

    @torch.no_grad()
    def predict(self, images):
        cls_out, box_out = self(images)
        probs = F.softmax(cls_out, dim=-1)
        return [postprocess(self.decode(box_out[i]) * IMG_SIZE, probs[i][:, 1:])
                for i in range(len(images))]



class YOLO(nn.Module):
    PRIORS = [(0.2, 0.2), (0.4, 0.4), (0.7, 0.7)]    # box shapes (w, h) as image fractions

    def __init__(self, num_classes):
        super().__init__()
        self.C = num_classes - 1                      # real classes only (no background)
        self.B = len(self.PRIORS)
        self.S = IMG_SIZE // 32
        self.backbone = nn.Sequential(
            conv_block(3, 32), nn.MaxPool2d(2),
            conv_block(32, 64), nn.MaxPool2d(2),
            conv_block(64, 128), nn.MaxPool2d(2),
            conv_block(128, 256), nn.MaxPool2d(2),
            conv_block(256, 512), nn.MaxPool2d(2),    # stride 32
        )
        self.head = nn.Conv2d(512, self.B * (5 + self.C), 1)
        self.register_buffer("priors", torch.tensor(self.PRIORS, dtype=torch.float32))

    def forward(self, x):
        out = self.head(self.backbone(x))             # (n, B*(5+C), S, S)
        return out.view(x.size(0), self.B, 5 + self.C, self.S, self.S).permute(0, 3, 4, 1, 2)
        # -> (n, S, S, B, 5+C)

    def compute_loss(self, images, boxes_list, labels_list):
        out = self(images)
        n, S, B = images.size(0), self.S, self.B
        obj = torch.zeros(n, S, S, B, dtype=torch.bool, device=images.device)
        t_box = torch.zeros(n, S, S, B, 4, device=images.device)
        t_cls = torch.zeros(n, S, S, B, dtype=torch.long, device=images.device)

        for i, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
            gt = boxes / IMG_SIZE
            cx, cy = (gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2
            w, h = (gt[:, 2] - gt[:, 0]).clamp(min=1e-3), (gt[:, 3] - gt[:, 1]).clamp(min=1e-3)
            gx, gy = (cx * S).clamp(0, S - 1e-4), (cy * S).clamp(0, S - 1e-4)
            col, row = gx.long(), gy.long()               # holds the box centre

            pw, ph = self.priors[:, 0], self.priors[:, 1]    # prior with the most similar shape
            inter = torch.min(w[:, None], pw[None]) * torch.min(h[:, None], ph[None])
            union = (w * h)[:, None] + (pw * ph)[None] - inter
            a = (inter / union).argmax(dim=1)

            obj[i, row, col, a] = True
            t_box[i, row, col, a] = torch.stack([gx - col, gy - row,
                                                 torch.log(w / pw[a]), torch.log(h / ph[a])], dim=1)
            t_cls[i, row, col, a] = labels - 1

        pred_xy = torch.sigmoid(out[..., 0:2])
        pred_wh = out[..., 2:4]
        conf = out[..., 4]
        cls_logits = out[..., 5:]

        box_loss = (F.mse_loss(pred_xy[obj], t_box[obj][:, :2], reduction="sum")
                    + F.mse_loss(pred_wh[obj], t_box[obj][:, 2:], reduction="sum"))
        conf_obj = F.binary_cross_entropy_with_logits(conf[obj], torch.ones_like(conf[obj]), reduction="sum")
        conf_noobj = F.binary_cross_entropy_with_logits(conf[~obj], torch.zeros_like(conf[~obj]), reduction="sum")
        cls_loss = F.cross_entropy(cls_logits[obj], t_cls[obj], reduction="sum")
        return (5.0 * box_loss + conf_obj + 0.5 * conf_noobj + cls_loss) / n

    @torch.no_grad()
    def predict(self, images):
        out = self(images)
        n, S = images.size(0), self.S
        gy, gx = torch.meshgrid(torch.arange(S, device=out.device),
                                torch.arange(S, device=out.device), indexing="ij")
        cx = (torch.sigmoid(out[..., 0]) + gx[None, :, :, None]) / S
        cy = (torch.sigmoid(out[..., 1]) + gy[None, :, :, None]) / S
        w = self.priors[:, 0] * torch.exp(out[..., 2].clamp(max=4))
        h = self.priors[:, 1] * torch.exp(out[..., 3].clamp(max=4))
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1) * IMG_SIZE
        scores = torch.sigmoid(out[..., 4:5]) * F.softmax(out[..., 5:], dim=-1)   # conf x class prob
        return [postprocess(boxes[i].reshape(-1, 4), scores[i].reshape(-1, self.C)) for i in range(n)]



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
    for c in range(1, num_classes):                     # class 0 is background
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



def random_flip(img, boxes):
    if random.random() < 0.5:
        img = img.flip(-1)
        boxes = boxes.clone()
        boxes[:, [0, 2]] = IMG_SIZE - boxes[:, [2, 0]]
    return img, boxes


def make_batch(batch, flip=False):
    imgs, boxes, labels = [], [], []
    for img, b, l in batch:
        if flip:
            img, b = random_flip(img, b)
        imgs.append(img)
        boxes.append(b)
        labels.append(l)
    return torch.stack(imgs), boxes, labels


def train_one_epoch(model, train, optimizer):
    model.train()
    random.shuffle(train)
    total = 0
    for i in range(0, len(train), BATCH_SIZE):
        batch = train[i:i + BATCH_SIZE]
        images, boxes, labels = make_batch(batch, flip=FLIP_AUGMENT)
        loss = model.compute_loss(images, boxes, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
    return total / len(train)


@torch.no_grad()
def validation_loss(model, val):
    model.eval()
    total = 0
    for i in range(0, len(val), BATCH_SIZE):
        batch = val[i:i + BATCH_SIZE]
        images, boxes, labels = make_batch(batch)
        total += model.compute_loss(images, boxes, labels).item() * len(batch)
    return total / len(val)


@torch.no_grad()
def evaluate(model, samples, num_classes):
    model.eval()
    predictions, ground_truths = [], []
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]
        images, boxes, labels = make_batch(batch)
        for (b, s, l), gt_b, gt_l in zip(model.predict(images), boxes, labels):
            predictions.append({"boxes": b, "scores": s, "labels": l})
            ground_truths.append({"boxes": gt_b.tolist(), "labels": gt_l.tolist()})
    return calculate_map(predictions, ground_truths, num_classes), predictions, ground_truths


def train_model(name, train, val, num_classes):
    model = {"ssd": SSD, "yolo": YOLO}[name](num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"\n=== {name} | parameters: {sum(p.numel() for p in model.parameters()):,} ===")

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

        # early stopping on validation mAP
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



def show_curves(name, hist, best_epoch):
    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, (ax_loss, ax_map) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, hist["train_loss"], label="Train loss")
    ax_loss.plot(epochs, hist["val_loss"], label="Validation loss")
    ax_loss.set_title(f"{name}: loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    if COMPUTE_TRAIN_MAP:
        ax_map.plot(epochs, hist["train_map"], label="Train mAP@0.5")
    ax_map.plot(epochs, hist["val_map"], label="Validation mAP@0.5")
    ax_map.set_title(f"{name}: mAP")
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("mAP@0.5")

    for ax in (ax_loss, ax_map):
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
        ax.legend()
    plt.tight_layout()
    plt.show()


def show_predictions(title, test, predictions, class_names, score_thresh=0.3):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (img, boxes, _), p in zip(axes.flat, test, predictions):
        ax.imshow(img.cpu().permute(1, 2, 0).numpy())
        for b in boxes.tolist():                                      # ground truth = green
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, color="lime", linewidth=2))
        for b, s, l in zip(p["boxes"], p["scores"], p["labels"]):     # predictions = red
            if s >= score_thresh:
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                           fill=False, color="red", linewidth=2))
                ax.text(b[0], b[1], f"{class_names[l - 1]} {s:.2f}", color="white",
                        fontsize=8, backgroundcolor="red")
    plt.suptitle(f"{title}  (green = ground truth, red = prediction)")
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

    results = {}
    for name in MODELS:                                     # same split for every model
        model, hist, best_epoch = train_model(name, train, val, num_classes)
        show_curves(name, hist, best_epoch)

        test_map, predictions, ground_truths = evaluate(model, test, num_classes)
        pr = [calculate_precision_recall(p["boxes"], p["scores"], p["labels"],
                                         g["boxes"], g["labels"])
              for p, g in zip(predictions, ground_truths)]
        precision, recall = np.mean([x[0] for x in pr]), np.mean([x[1] for x in pr])
        results[name] = (test_map, precision, recall)
        print(f"{name} TEST  mAP@0.5: {test_map:.3f} | precision: {precision:.3f} "
              f"| recall: {recall:.3f}")
        show_predictions(name, test, predictions, class_names)

    print("\nModel comparison (test set)")
    print(f"{'model':<12}{'mAP@0.5':>10}{'precision':>12}{'recall':>10}")
    for name, (m, p, r) in results.items():
        print(f"{name:<12}{m:>10.3f}{p:>12.3f}{r:>10.3f}")


if __name__ == "__main__":
    main()
