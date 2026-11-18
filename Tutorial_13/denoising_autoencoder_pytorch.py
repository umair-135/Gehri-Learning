import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

import matplotlib.pyplot as plt


DATA_ROOT = "data"          
EPOCHS = 10                 
BATCH_SIZE = 128
LR = 1e-3
NOISE_FACTOR = 0.5
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DenoisingAutoencoder(nn.Module):
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


class DeepAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 14 x 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 7 x 7
            nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
            nn.Linear(256, latent_dim), nn.ReLU(),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),     # 14 x 14
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),     # 28 x 28
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.decoder_fc(self.encoder(x)).view(-1, 64, 7, 7))


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 14 x 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 7 x 7
            nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(),
                                        nn.Linear(256, 64 * 7 * 7), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )

    def decode(self, z):
        return self.decoder(self.decoder_fc(z).view(-1, 64, 7, 7))

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)      # reparameterization trick
        return self.decode(z), mu, logvar


def compute_loss(model, noisy, clean):
    if isinstance(model, VariationalAutoencoder):
        recon, mu, logvar = model(noisy)
        bce = F.binary_cross_entropy(recon, clean, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (bce + kl) / len(noisy)
    return F.binary_cross_entropy(model(noisy), clean, reduction="sum") / len(noisy)


@torch.no_grad()
def denoise(model, noisy):
    model.eval()
    out = model(noisy)
    return out[0] if isinstance(model, VariationalAutoencoder) else out


def add_noise(images):
    return (images + NOISE_FACTOR * torch.randn_like(images)).clamp(0.0, 1.0)


def load_data():
    train = datasets.MNIST(DATA_ROOT, train=True, download=True)
    test = datasets.MNIST(DATA_ROOT, train=False, download=True)
    x_train = (train.data.float() / 255.0).unsqueeze(1)      
    x_test = (test.data.float() / 255.0).unsqueeze(1)
    return x_train, x_test


def make_loaders(x_train, x_test):
    train_noisy, test_noisy = add_noise(x_train), add_noise(x_test)  
    train_loader = DataLoader(TensorDataset(train_noisy, x_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_noisy, x_test), batch_size=BATCH_SIZE)
    return train_loader, test_loader, test_noisy

def run_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    total, n = 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for noisy, clean in loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            loss = compute_loss(model, noisy, clean)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(noisy)
            n += len(noisy)
    return total / n


def train_model(name, model, train_loader, test_loader):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"\n=== {name} | parameters: {sum(p.numel() for p in model.parameters()):,} ===")

    train_losses, val_losses = [], []
    for epoch in range(1, EPOCHS + 1):
        train_losses.append(run_epoch(model, train_loader, optimizer))
        val_losses.append(run_epoch(model, test_loader))
        print(f"Epoch {epoch:2d} | train loss {train_losses[-1]:.2f} | val loss {val_losses[-1]:.2f}")
    return model, train_losses, val_losses


def show_curves(histories):
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4), squeeze=False)
    for ax, (name, (train_losses, val_losses)) in zip(axes[0], histories.items()):
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="Train loss")
        ax.plot(epochs, val_losses, label="Validation loss")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss per image")
        ax.legend()
    plt.tight_layout()
    plt.show()


def show_denoised(models, x_test, test_noisy, n=10):
    noisy = test_noisy[:n].to(DEVICE)
    rows = [("Clean", x_test[:n]), ("Noisy", noisy.cpu())]
    rows += [(name, denoise(m, noisy).cpu()) for name, m in models.items()]

    fig, axes = plt.subplots(len(rows), n, figsize=(2 * n, 2 * len(rows)))
    for r, (title, imgs) in enumerate(rows):
        for i in range(n):
            axes[r, i].imshow(imgs[i, 0], cmap="gray")
            axes[r, i].axis("off")
        axes[r, 0].set_title(title, loc="left", fontsize=10)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_vae_samples(vae, n=10):
    vae.eval()
    z = torch.randn(n, vae.latent_dim, device=DEVICE)     
    imgs = vae.decode(z).cpu()
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
    for i in range(n):
        axes[i].imshow(imgs[i, 0], cmap="gray")
        axes[i].axis("off")
    plt.suptitle("New images generated by the VAE from random latent vectors")
    plt.tight_layout()
    plt.show()


def main():
    torch.manual_seed(SEED)
    x_train, x_test = load_data()
    train_loader, test_loader, test_noisy = make_loaders(x_train, x_test)

    candidates = {"Denoising AE": DenoisingAutoencoder(),
                  "Deep AE": DeepAutoencoder(),
                  "VAE": VariationalAutoencoder()}
    models, histories = {}, {}
    for name, model in candidates.items():
        models[name], *hist = train_model(name, model, train_loader, test_loader)
        histories[name] = tuple(hist)

    show_curves(histories)
    show_denoised(models, x_test, test_noisy)
    show_vae_samples(models["VAE"])

    print("\nTest MSE between denoised and clean images (lower is better)")
    noisy = test_noisy.to(DEVICE)
    print(f"{'Noisy input':<14} {F.mse_loss(noisy.cpu(), x_test).item():.5f}")
    for name, model in models.items():
        print(f"{name:<14} {F.mse_loss(denoise(model, noisy).cpu(), x_test).item():.5f}")

if __name__ == "__main__":
    main()
