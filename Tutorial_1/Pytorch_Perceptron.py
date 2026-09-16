import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class PerceptronTorch(nn.Module):
    def __init__(self, input_dim):
        super(PerceptronTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        z = self.linear(x)
        return torch.sigmoid(z)

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    header=None,
)
df = shuffle(df)

X = df.iloc[:, 0:4].values.astype(np.float32)
y = df.iloc[:, 4].values

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25)

train_labels = np.where(train_labels == "Iris-setosa", 1, 0).astype(np.float32)
test_labels = np.where(test_labels == "Iris-setosa", 1, 0).astype(np.float32)

X_train = torch.from_numpy(train_data)
y_train = torch.from_numpy(train_labels).view(-1, 1)  # column vector, shape (N, 1)
X_test = torch.from_numpy(test_data)
y_test = torch.from_numpy(test_labels).view(-1, 1)

model = PerceptronTorch(input_dim=4)

criterion = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

n_iter = 100

for epoch in range(n_iter):
    optimizer.zero_grad()          
    outputs = model(X_train)      
    loss = criterion(outputs, y_train)  
    loss.backward()               
    optimizer.step()               

    print(f"Epoch {epoch + 1}/{n_iter} - Loss: {loss.item():.4f}")

with torch.no_grad():  
    test_outputs = model(X_test)
    test_preds = (test_outputs >= 0.5).float()  

accuracy = accuracy_score(y_test.numpy(), test_preds.numpy())
print("Accuracy:", round(accuracy, 2) * 100, "%")

print("\nEnter flower measurements to predict the species:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

manual_input = torch.tensor([[sepal_length, sepal_width, petal_length, petal_width]], dtype=torch.float32)

with torch.no_grad():
    manual_pred = model(manual_input)

if manual_pred.item() >= 0.5:
    print("Prediction: Iris-setosa")
else:
    print("Prediction: Not Iris-setosa")