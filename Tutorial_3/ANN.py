import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

torch.manual_seed(42)  # for reproducible results across runs

# transforms.ToTensor() converts images to tensors AND scales pixel values from 0-255 to 0-1 
transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# PyTorch's CrossEntropyLoss expects plain integer class labels (0-9) and computes the equivalent of
# categorical cross-entropy internally, including the softmax.


# Validation split
val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class MNIST_ANN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(MNIST_ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 128)
        self.act3 = nn.GELU()
        self.drop3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(128, 64)
        self.act4 = nn.GELU()
        self.drop4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(64, 32)
        self.act5 = nn.GELU()
        self.drop5 = nn.Dropout(dropout_rate)
        self.fc6 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.drop1(self.act1(self.fc1(x)))
        x = self.drop2(self.act2(self.fc2(x)))
        x = self.drop3(self.act3(self.fc3(x)))
        x = self.drop4(self.act4(self.fc4(x)))
        x = self.drop5(self.act5(self.fc5(x)))
        x = self.fc6(x)  # no dropout on the output layer
        return x

model = MNIST_ANN()
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


# Training the model
def evaluate(loader):
    """Compute average loss and accuracy over a given DataLoader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


n_epochs = 10
history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

# Early stopping setup
patience = 3               # how many epochs to tolerate no improvement
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

for epoch in range(n_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    val_loss, val_acc = evaluate(val_loader)

    history["loss"].append(train_loss)
    history["accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    print(f"Epoch {epoch + 1}/{n_epochs} - "
          f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
          f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

     # --- Early stopping check ---
    if val_loss < best_val_loss - 1e-4:  # meaningful improvement
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"    No improvement for {patience_counter} epoch(s).")
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

# Restore the best weights seen during training (not necessarily the last epoch's)
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Restored model weights from the epoch with the lowest validation loss.")
    
# Evaluating the model on test data
test_loss, test_accuracy = evaluate(test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Visualizing training and validation performance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Making predictions and visualizing results
model.eval()

# Picking one batch of test images to visualize
test_images, test_labels = next(iter(test_loader))
test_images_device = test_images.to(device)

with torch.no_grad():
    logits = model(test_images_device)
    probs = torch.softmax(logits, dim=1)   # convert logits to probabilities, for display only
    predictions = torch.argmax(probs, dim=1).cpu()

# Display the first test image with its true and predicted label
plt.figure(figsize=(5, 5))
plt.imshow(test_images[0].squeeze(), cmap="gray")
plt.title(f"True Label: {test_labels[0].item()}, Predicted: {predictions[0].item()}")
plt.axis("off")
plt.show()

# Display a 3x3 grid of test images with true/predicted labels
num_images = 9
plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i].squeeze(), cmap="gray")
    plt.title(f"True: {test_labels[i].item()}, Predicted: {predictions[i].item()}")
    plt.axis("off")
plt.tight_layout()
plt.show()