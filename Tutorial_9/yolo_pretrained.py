import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

import matplotlib.pyplot as plt


DATA_DIR = "/home/umair/Deep_Learning/data/Dataset_Detection"
WORK_DIR = "yolo_work"     # recreated on every run
MODEL_WEIGHTS = "yolov8n.pt"  
IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 8
PATIENCE = 10            
SEED = 0
DEVICE = 0 if torch.cuda.is_available() else "cpu"


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


def prepare_dataset(folder, work_dir):
    raw = []
    for jf in sorted(Path(folder).glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        if len(data["shapes"]) > 0:
            raw.append((jf, data))

    class_names = sorted({s["label"].strip() for _, d in raw for s in d["shapes"]})
    class_to_idx = {n: i + 1 for i, n in enumerate(class_names)}    

    samples = []                                        
    for jf, data in raw:
        img = Image.open(jf.parent / Path(data["imagePath"]).name)
        img = ImageOps.exif_transpose(img).convert("RGB")          
        w, h = img.size
        boxes, labels = [], []
        for s in data["shapes"]:
            pts = np.array(s["points"], dtype=np.float32)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            x1, x2 = float(np.clip(x1, 0, w)), float(np.clip(x2, 0, w))
            y1, y2 = float(np.clip(y1, 0, h)), float(np.clip(y2, 0, h))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(class_to_idx[s["label"].strip()])
        if len(boxes) == 0:
            continue
        samples.append((img, jf.stem, boxes, labels))

    random.shuffle(samples)
    n_train, n_val = int(0.70 * len(samples)), int(0.15 * len(samples))
    split_samples = {"train": samples[:n_train],
                     "val": samples[n_train:n_train + n_val],
                     "test": samples[n_train + n_val:]}

    work_dir = Path(work_dir).resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)                         
    splits = {}
    for split, items in split_samples.items():
        (work_dir / "images" / split).mkdir(parents=True)
        (work_dir / "labels" / split).mkdir(parents=True)
        splits[split] = []
        for img, stem, boxes, labels in items:
            w, h = img.size
            img_path = work_dir / "images" / split / f"{stem}.jpg"
            img.save(img_path, quality=95)
            with open(work_dir / "labels" / split / f"{stem}.txt", "w") as f:
                for (x1, y1, x2, y2), l in zip(boxes, labels):
                    f.write(f"{l - 1} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")     
            splits[split].append((str(img_path), boxes, labels))

    yaml_path = work_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {work_dir}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
        for i, n in enumerate(class_names):
            f.write(f"  {i}: {n}\n")
    return splits, class_names, yaml_path


def show_curves(results_csv):
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    epochs = df["epoch"]
    train_cols = [c for c in df.columns if c.startswith("train/") and c.endswith("_loss")]
    val_cols = [c for c in df.columns if c.startswith("val/") and c.endswith("_loss")]
    train_loss = df[train_cols].sum(axis=1)
    val_loss = df[val_cols].sum(axis=1)
    fitness = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]   
    best_epoch = int(epochs[fitness.idxmax()])

    fig, (ax_loss, ax_map) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(epochs, train_loss, label="Train loss")
    ax_loss.plot(epochs, val_loss, label="Validation loss")
    ax_loss.set_title("YOLO: loss (sum of components)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    ax_map.plot(epochs, df["metrics/mAP50(B)"], label="Validation mAP@0.5")
    ax_map.set_title("YOLO: validation mAP")
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
    for ax, (img_path, boxes, _), p in zip(axes.flat, test, predictions):
        ax.imshow(Image.open(img_path))
        for b in boxes:                                               # ground truth = green
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, color="lime", linewidth=2))
        for b, s, l in zip(p["boxes"], p["scores"], p["labels"]):     # predictions = red
            if s >= score_thresh:
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                           fill=False, color="red", linewidth=2))
                ax.text(b[0], b[1], f"{class_names[l - 1]} {s:.2f}", color="white",
                        fontsize=8, backgroundcolor="red")
    plt.suptitle("YOLO (pretrained)  (green = ground truth, red = prediction)")
    plt.show()


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    splits, class_names, yaml_path = prepare_dataset(DATA_DIR, WORK_DIR)
    num_classes = len(class_names) + 1
    print(f"classes: {class_names}")
    print(f"train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")

    model = YOLO(MODEL_WEIGHTS)
    model.train(data=str(yaml_path), epochs=EPOCHS, patience=PATIENCE, imgsz=IMG_SIZE,
                batch=BATCH_SIZE, device=DEVICE, seed=SEED,
                project=str(Path(WORK_DIR).resolve() / "runs"), name="train", exist_ok=True,
                plots=False)                              # we draw our own figures below
    save_dir = Path(model.trainer.save_dir)
    show_curves(save_dir / "results.csv")
    
    best = YOLO(save_dir / "weights" / "best.pt")
    test = splits["test"]
    results = best.predict([t[0] for t in test], conf=0.05, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    predictions = [{"boxes": r.boxes.xyxy.tolist(), "scores": r.boxes.conf.tolist(),
                    "labels": (r.boxes.cls + 1).int().tolist()} for r in results]   # back to 1-based ids
    ground_truths = [{"boxes": boxes, "labels": labels} for _, boxes, labels in test]

    test_map = calculate_map(predictions, ground_truths, num_classes)
    pr = [calculate_precision_recall(p["boxes"], p["scores"], p["labels"], g["boxes"], g["labels"])
          for p, g in zip(predictions, ground_truths)]
    print(f"\nTEST  mAP@0.5: {test_map:.3f} | precision: {np.mean([x[0] for x in pr]):.3f} "
          f"| recall: {np.mean([x[1] for x in pr]):.3f}")
    show_predictions(test, predictions, class_names)


if __name__ == "__main__":
    main()