import torch                                   
import torch.nn as nn                          
import torch.optim as optim                    
from torch.utils.data import DataLoader        
import torchvision                             
import torchvision.transforms as transforms
import matplotlib.pyplot as plt                
import numpy as np                             

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


plt.figure(figsize=(10, 20))
for i in range(5):
    img, label = train_dataset[i]
    plt.subplot(5, 1, i + 1)
    plt.xticks([])                                  
    plt.yticks([])                                  
    plt.grid(False)                                
    plt.imshow(img.permute(1, 2, 0).numpy())        
    plt.xlabel(class_names[label])                  
plt.show()


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Input (3x32x32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(64 * 1 * 1, 64),
            nn.LazyLinear(256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5), 
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.5),
            
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = CNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Architecture summary
print(model)

def run_epoch(loader, train=True):
    model.train(train)
    running_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)                  
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()              
                loss.backward()                      
                optimizer.step()                     

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


epochs = 10
history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

for epoch in range(epochs):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(test_loader, train=False)

    history['loss'].append(train_loss)
    history['accuracy'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)

    print(f"Epoch {epoch + 1}/{epochs} - "
          f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
          f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")


test_loss, test_acc = run_epoch(test_loader, train=False)
print(f"Test accuracy: {test_acc}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)                                  
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)                                 
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.eval()
test_images = torch.stack([test_dataset[i][0] for i in range(5)])
test_labels = np.array([test_dataset[i][1] for i in range(5)])

with torch.no_grad():
    logits = model(test_images.to(device))
    predictions = torch.softmax(logits, dim=1).cpu().numpy()   

def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)                                   
    plt.xticks([])                                    
    plt.yticks([])                                   
    plt.imshow(img.permute(1, 2, 0).numpy())          

    predicted_label = np.argmax(predictions_array)

    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(f"Predicted: {class_names[predicted_label]} (True: {class_names[true_label]})",
               color=color)

plt.figure(figsize=(5, 10))
for i in range(5):
    plt.subplot(5, 1, i + 1)                          
    plot_image(i, predictions, test_labels, test_images, class_names)

plt.tight_layout()                                    
plt.show()