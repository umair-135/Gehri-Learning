import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

DATA_ROOT = "data"       
EPOCHS = 20              
BATCH_SIZE = 128
LR = 1e-3
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BasicAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),   
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),  
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_data():
    tf = transforms.ToTensor()                         
    train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(test, batch_size=BATCH_SIZE))


def run_epoch(model, loader, loss_fn, optimizer=None):
    model.train(optimizer is not None)
    total, n = 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, _ in loader:
            x = x.to(DEVICE)
            loss = loss_fn(model(x), x)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / n


def train_model(name, model, train_loader, test_loader):
    model = model.to(DEVICE)
    loss_fn = nn.BCELoss()                           
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"\n=== {name} | parameters: {sum(p.numel() for p in model.parameters()):,} ===")

    train_losses, val_losses = [], []
    for epoch in range(1, EPOCHS + 1):
        train_losses.append(run_epoch(model, train_loader, loss_fn, optimizer))
        val_losses.append(run_epoch(model, test_loader, loss_fn))
        print(f"Epoch {epoch:2d} | train loss {train_losses[-1]:.4f} | val loss {val_losses[-1]:.4f}")
    return model, train_losses, val_losses


def show_curves(histories):
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4), squeeze=False)
    for ax, (name, (train_losses, val_losses)) in zip(axes[0], histories.items()):
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="Train loss")
        ax.plot(epochs, val_losses, label="Validation loss")
        ax.set_title(f"{name}: reconstruction loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE loss")
        ax.legend()
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_reconstructions(models, test_loader, n=10):
    x, _ = next(iter(test_loader))
    x = x[:n].to(DEVICE)
    rows = [("Original", x)] + [(name, m.eval()(x)) for name, m in models.items()]

    fig, axes = plt.subplots(len(rows), n, figsize=(2 * n, 2 * len(rows)))
    for r, (title, imgs) in enumerate(rows):
        for i in range(n):
            axes[r, i].imshow(imgs[i, 0].cpu(), cmap="gray")
            axes[r, i].axis("off")
        axes[r, 0].set_title(title, loc="left", fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    torch.manual_seed(SEED)
    train_loader, test_loader = load_data()

    candidates = {"Autoencoder": BasicAutoencoder()}
    models, histories = {}, {}
    for name, model in candidates.items():
        models[name], *hist = train_model(name, model, train_loader, test_loader)
        histories[name] = tuple(hist)

    show_curves(histories)
    show_reconstructions(models, test_loader)

    mse = nn.MSELoss()                                
    print("\nFinal test reconstruction error")
    for name, model in models.items():
        print(f"{name:<18} MSE: {run_epoch(model, test_loader, mse):.5f}")


if __name__ == "__main__":
    main()
