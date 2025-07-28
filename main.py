# %%cell 1
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# %%cell 2
df = pd.read_csv("f1_pitstop_dataset_processed.csv")
df.head()

# %%cell 3
df["will_pit_next_lap"].value_counts()

# %%cell 4
print("NaN values in the dataset:")
print(df.isna().sum())

df = df.dropna()
print("NaN values after dropping rows:")
print(df.isna().sum())
# %%cell 11
df["will_pit_next_lap"].value_counts()


# %%cell 5
X = df.drop(columns=["will_pit_next_lap"])
y = df["will_pit_next_lap"]

scalar = preprocessing.MinMaxScaler()
X = pd.DataFrame(scalar.fit_transform(X), columns=X.columns)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=32
)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
y_train = pd.Series(y_train, name=y.name)
y_test = pd.Series(y_test, name=y.name)


# %%cell 6
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


# %%cell 6
class PitstopModel(torch.nn.Module):
    def __init__(self, input_size):
        super(PitstopModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


# %%cell 7
# Calculate class weights to handle imbalance
pos_weight = torch.tensor([len(y_train[y_train == 0]) / len(y_train[y_train == 1])])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
model = PitstopModel(X_train_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%cell 8
epochs = 600
for epoch in range(epochs):
    y_pred = model(X_train_tensor)

    loss = criterion(y_pred.squeeze(), y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# %%cell 9

with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_binary = (y_pred.squeeze() > 0.5).float()  # Convert probabilities

    # Convert tensors to numpy for sklearn metrics
    y_pred_np = y_pred_binary.numpy()
    y_test_np = y_test_tensor.numpy()

    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    accuracy = accuracy_score(y_test_np, y_pred_np)
    precision = precision_score(
        y_test_np, y_pred_np, average="weighted", labels=np.unique(y_pred)
    )
    recall = recall_score(
        y_test_np, y_pred_np, average="weighted", labels=np.unique(y_pred)
    )
    f1 = f1_score(y_test_np, y_pred_np, average="weighted", labels=np.unique(y_pred))
    cm = confusion_matrix(y_test_np, y_pred_np)

    # Print metrics
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Pit", "Pit"],
        yticklabels=["No Pit", "Pit"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
