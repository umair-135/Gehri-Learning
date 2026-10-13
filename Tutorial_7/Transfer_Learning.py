import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

EXPERIMENTS = [
    ("vgg16",    "fine_tuning"),
    ("resnet50", "feature_extraction"),
]

EPOCHS          = 15        # maximum number of epochs
BATCH_SIZE      = 64
LR_HEAD         = 1e-3      # learning rate for the NEW classification layers
LR_BACKBONE     = 1e-5      # learning rate for UNFROZEN pretrained layers (fine-tuning only)
WEIGHT_DECAY    = 1e-4      # L2 regularisation


UNFREEZE_LAYERS = 7         # number of LAST conv layers to unfreeze (fine_tuning mode only)

HEAD_HIDDEN_UNITS = 512     # New classification head (replaces the ImageNet 1000-class layer)
DROPOUT           = 0.5

EARLY_STOPPING = True
PATIENCE       = 3          # epochs to wait before stopping

DATASET_PERCENT = 100       # % of the 50,000 CIFAR-10 training images
VAL_PERCENT     = 20        # % of the used training images
TEST_PERCENT    = 100       # % of the 10,000 test images used for the final evaluation
IMG_SIZE        = 256       # images are resized to IMG_SIZE x IMG_SIZE
AUGMENTATION    = False      # random horizontal flip on training images

SEED       = 42
PLOT_PATH  = "tutorial7_curves.png"   # set to None to skip saving the plot
SHOW_PLOT  = True                   # True = also open the plot window


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10



def get_dataloaders():
    resize = [transforms.Resize((IMG_SIZE, IMG_SIZE))] if IMG_SIZE != 32 else []

    train_tf = transforms.Compose(
        resize + ([transforms.RandomHorizontalFlip()] if AUGMENTATION else [])
        + [transforms.ToTensor()])
    eval_tf = transforms.Compose(resize + [transforms.ToTensor()])

    full_train = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    full_val = datasets.CIFAR10("./data", train=True, download=True, transform=eval_tf)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=eval_tf)

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(full_train), generator=g).tolist()
    n_used = max(1, int(len(full_train) * DATASET_PERCENT / 100))
    n_val = int(n_used * VAL_PERCENT / 100)
    n_train = n_used - n_val
    train_idx, val_idx = perm[:n_train], perm[n_train:n_used]

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx) if n_val > 0 else None
    n_test = max(1, int(len(test_ds) * TEST_PERCENT / 100))
    test_ds = Subset(test_ds, range(n_test))

    kw = dict(batch_size=BATCH_SIZE, pin_memory=DEVICE.type == "cuda")
    print(f"Data: {n_train} train / {n_val} val / {n_test} test images, "
          f"{IMG_SIZE}x{IMG_SIZE}")
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw) if val_ds else None,
            DataLoader(test_ds, shuffle=False, **kw))


def make_head(in_features):
    layers = []
    if HEAD_HIDDEN_UNITS > 0:
        layers += [nn.Linear(in_features, HEAD_HIDDEN_UNITS), nn.ReLU(inplace=True)]
        if DROPOUT > 0:
            layers.append(nn.Dropout(DROPOUT))
        layers.append(nn.Linear(HEAD_HIDDEN_UNITS, NUM_CLASSES))
    else:
        if DROPOUT > 0:
            layers.append(nn.Dropout(DROPOUT))
        layers.append(nn.Linear(in_features, NUM_CLASSES))
    return nn.Sequential(*layers)


def build_model(arch, mode):
    """Returns (model, backbone_params_to_finetune, new_head_params)."""
    if arch == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier = make_head(512 * 7 * 7)
        head = model.classifier
        conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = make_head(model.fc.in_features)
        head = model.fc
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    backbone_params = []
    if mode == "fine_tuning":
        for conv in conv_layers[-UNFREEZE_LAYERS:]:
            for p in conv.parameters():
                p.requires_grad = True
                backbone_params.append(p)
    elif mode != "feature_extraction":
        raise ValueError(f"Unknown mode: {mode}")

    return model.to(DEVICE), backbone_params, list(head.parameters())


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def run_epoch(model, loader, criterion, optimizer=None, freeze_bn=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:               
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def train_and_evaluate(title, arch, mode, loaders):
    train_loader, val_loader, test_loader = loaders
    model, backbone_params, head_params = build_model(arch, mode)
    trainable, total = count_params(model)
    print(f"\n===== {title} =====")
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    groups = [{"params": head_params, "lr": LR_HEAD}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": LR_BACKBONE})
    optimizer = torch.optim.Adam(groups, weight_decay=WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1))
    criterion = nn.CrossEntropyLoss()
    freeze_bn = (mode == "feature_extraction")

    hist = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_score, best_state, wait = -1.0, None, 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, freeze_bn)
        msg = f"Epoch {epoch:2d}/{EPOCHS}  train loss {tr_loss:.4f} acc {tr_acc:.3f}"
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)

        if val_loader:
            va_loss, va_acc = run_epoch(model, val_loader, criterion)
            hist["val_loss"].append(va_loss); hist["val_acc"].append(va_acc)
            msg += f" | val loss {va_loss:.4f} acc {va_acc:.3f}"
            score = va_acc
            if scheduler:
                scheduler.step(va_loss)
        else:
            score = tr_acc
        print(msg)

        if score > best_score:
            best_score, wait = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if EARLY_STOPPING and wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best {'val' if val_loader else 'train'} acc {best_score:.3f})")
                break

    if EARLY_STOPPING and best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, test_loader, criterion)
    elapsed = time.time() - t0
    print(f"{title} -> TEST loss {test_loss:.4f} | acc {test_acc * 100:.2f}% | time {elapsed:.0f}s")

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"hist": hist, "best": best_score, "test_acc": test_acc,
            "trainable": trainable, "time": elapsed}


def plot_results(results):
    try:
        import matplotlib
        if not SHOW_PLOT:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot (pip install matplotlib)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, r in results.items():
        h = r["hist"]
        if h["val_acc"]:
            axes[0].plot(h["val_acc"], label=f"{name} (val)")
            axes[1].plot(h["val_loss"], label=f"{name} (val)")
        axes[0].plot(h["train_acc"], "--", label=f"{name} (train)")
        axes[1].plot(h["train_loss"], "--", label=f"{name} (train)")
    axes[0].set_title("Accuracy"); axes[1].set_title("Loss")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    if PLOT_PATH:
        fig.savefig(PLOT_PATH, dpi=150)
        print(f"Saved training curves to {PLOT_PATH}")
    if SHOW_PLOT:
        plt.show()


def main():
    torch.manual_seed(SEED)
    print(f"Using device: {DEVICE}")
    loaders = get_dataloaders()

    results = {}
    for arch, mode in EXPERIMENTS:
        label = f"{arch.upper()} {mode.replace('_', ' ')}"
        if mode == "fine_tuning":
            label += f" (last {UNFREEZE_LAYERS} conv)"
        results[label] = train_and_evaluate(label, arch, mode, loaders)

    print("\n" + "=" * 92)
    print(f"{'Experiment':46s}{'Trainable':>12s}{'Val acc':>10s}{'Test acc':>10s}{'Time':>8s}")
    for name, r in results.items():
        val = f"{r['best'] * 100:.2f}%" if loaders[1] else "n/a"
        print(f"{name:46s}{r['trainable']:>12,}{val:>10s}{r['test_acc'] * 100:>9.2f}%{r['time']:>7.0f}s")
    plot_results(results)


if __name__ == "__main__":
    main()