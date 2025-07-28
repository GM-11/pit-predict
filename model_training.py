# %%cell 1
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.utils import resample

torch.set_grad_enabled(True)

# %%cell 2
df = pd.read_csv("f1_pitstop_dataset_processed.csv")
df.head()

# %%cell 3
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using {device}")

# %%cell 4
print("NaN values in the dataset:")
print(df.isna().sum())

df = df.dropna()
print("NaN values after dropping rows:")
print(df.isna().sum())

# %%cell 5
print(df["will_pit_next_lap"].value_counts())

# Balance the dataset before training

# Separate classes
df_majority = df[df["will_pit_next_lap"] == 0]
df_minority = df[df["will_pit_next_lap"] == 1]

# Oversample minority to 10% ratio (instead of 3%)
target_size = len(df_majority) // 10  # 10% instead of 3%
df_minority_upsampled = resample(
    df_minority, replace=True, n_samples=target_size, random_state=42
)

# Make sure both dataframes are properly typed
df_majority = pd.DataFrame(df_majority)
df_minority_upsampled = pd.DataFrame(df_minority_upsampled)

# Use concat with explicit dataframes
df = pd.concat([df_majority, df_minority_upsampled], ignore_index=True)
print(f"New ratio: {df['will_pit_next_lap'].value_counts()}")


# %%cell 6
X = df.drop(columns=["will_pit_next_lap"])
y = df["will_pit_next_lap"]

# %%cell 7
scalar = preprocessing.MinMaxScaler()
X = pd.DataFrame(scalar.fit_transform(X), columns=X.columns)

X["tyre_age"] *= 3
X["lap_number"] *= 1.5
X["position"] *= 1.1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=32
)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
y_train = pd.Series(y_train, name=y.name)
y_test = pd.Series(y_test, name=y.name)
X_train

# %%cell 8
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)


# %%cell 9
class PitstopModel(torch.nn.Module):
    def __init__(self, input_size):
        super(PitstopModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.network(x)


# %%cell 10
class_0_count = (y_train_tensor == 0).sum().float()
class_1_count = (y_train_tensor == 1).sum().float()
weight_ratio = class_0_count / class_1_count
pos_weight = torch.tensor([weight_ratio * 0.2], dtype=torch.float32).to(device)
print(f"Positive class weight: {pos_weight.item():.2f}")

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
model = PitstopModel(X_train_tensor.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

# %%cell 11
epochs = 6000
losses = []
learning_rates = []

for epoch in range(epochs):
    # Training step
    model.train()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred.squeeze(), y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Step scheduler (different for each type)
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(loss)  # Use training loss for plateau detection
    else:
        scheduler.step()  # For StepLR and CosineAnnealingLR

    # Track metrics
    losses.append(loss.item())
    current_lr = optimizer.param_groups[0]["lr"]
    learning_rates.append(current_lr)

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
        )

        # Check if loss stopped improving (simple early stopping)
        if epoch > 200:
            recent_avg = sum(losses[-50:]) / 50
            older_avg = sum(losses[-150:-100]) / 50
            improvement = older_avg - recent_avg

            if improvement < 0.0001:  # Very small improvement
                print("Loss plateaued. Consider stopping or adjusting LR.")

# %%cell 12
plt.figure(figsize=(15, 5))

# Plot 1: Loss over time
plt.subplot(1, 3, 1)
plt.plot(losses, label="Training Loss", color="blue", alpha=0.7)
# Add smoothed line
if len(losses) > 100:
    smoothed = [
        sum(losses[max(0, i - 50) : i + 1]) / min(i + 1, 51) for i in range(len(losses))
    ]
    plt.plot(smoothed, label="Smoothed Loss", color="red", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Learning Rate over time
plt.subplot(1, 3, 2)
plt.plot(learning_rates, label="Learning Rate", color="green")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale("log")  # Log scale for better visualization

# Plot 3: Loss improvement rate
plt.subplot(1, 3, 3)
if len(losses) > 100:
    improvement_rate = []
    window = 100
    for i in range(window, len(losses)):
        old_avg = sum(losses[i - window : i - window + 50]) / 50
        new_avg = sum(losses[i - 50 : i]) / 50
        improvement_rate.append(old_avg - new_avg)

    plt.plot(range(window, len(losses)), improvement_rate, color="purple")
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Improvement Rate")
    plt.title("Learning Progress")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("images/training_progress_with_lr.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFinal Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
print(f"Final Loss: {losses[-1]:.4f}")
print(f"Best Loss: {min(losses):.4f} at epoch {losses.index(min(losses)) + 1}")

# %%cell 13
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_proba = torch.sigmoid(y_pred.squeeze())

    # Move to CPU for numpy conversion
    y_pred_proba_np = y_pred_proba.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    # Test multiple thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}

    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)
    print(
        f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'Pos_Pred':<8}"
    )
    print("-" * 70)

    results = []

    for threshold in thresholds:
        # Make predictions with current threshold
        y_pred_binary = (y_pred_proba_np > threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test_np, y_pred_binary)
        precision = precision_score(
            y_test_np, y_pred_binary, average="binary", zero_division=0.0
        )
        recall = recall_score(
            y_test_np, y_pred_binary, average="binary", zero_division=0.0
        )
        f1 = f1_score(y_test_np, y_pred_binary, average="binary", zero_division=0.0)

        # Count positive predictions
        pos_predictions = np.sum(y_pred_binary)
        total_actual_pos = np.sum(y_test_np)

        # Store results
        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pos_pred": pos_predictions,
                "y_pred": y_pred_binary,
            }
        )

        # Print row
        print(
            f"{threshold:<10.1f} {accuracy * 100:<9.2f}% {precision:<10.3f} {recall:<7.3f} {f1:<7.3f} {pos_predictions:<8}"
        )

        # Track best F1
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pos_pred": pos_predictions,
                "y_pred": y_pred_binary,
            }

    print("=" * 70)
    print(f"BEST PERFORMANCE: Threshold = {best_threshold}")
    print(f"Accuracy: {best_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(
        f"Positive Predictions: {best_metrics['pos_pred']} / {np.sum(y_test_np)} actual"
    )
    print("=" * 70)

    # Plot threshold analysis
    import matplotlib.pyplot as plt

    thresholds_list = [r["threshold"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]

    plt.figure(figsize=(12, 5))

    # Plot 1: Precision vs Recall vs F1
    plt.subplot(1, 2, 1)
    plt.plot(thresholds_list, precisions, "b-o", label="Precision", linewidth=2)
    plt.plot(thresholds_list, recalls, "r-s", label="Recall", linewidth=2)
    plt.plot(thresholds_list, f1s, "g-^", label="F1 Score", linewidth=2)
    plt.axvline(
        x=best_threshold,
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"Best F1 ({best_threshold})",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Metrics vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.05, 0.85)
    plt.ylim(0, 1)

    # Plot 2: Number of positive predictions
    plt.subplot(1, 2, 2)
    pos_preds = [r["pos_pred"] for r in results]
    plt.plot(thresholds_list, pos_preds, "purple", marker="o", linewidth=2)
    plt.axhline(
        y=np.sum(y_test_np),
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Actual Positives ({np.sum(y_test_np)})",
    )
    plt.axvline(
        x=best_threshold,
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"Best Threshold ({best_threshold})",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Number of Positive Predictions")
    plt.title("Positive Predictions vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.05, 0.85)

    plt.tight_layout()
    plt.savefig("images/threshold_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Confusion matrix for best threshold
    cm = confusion_matrix(y_test_np, best_metrics["y_pred"])

    plt.figure(figsize=(8, 6))
    import seaborn as sns

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
    plt.title(f"Confusion Matrix (Threshold = {best_threshold})")
    plt.savefig("images/confusion_matrix_best.png", dpi=300, bbox_inches="tight")
    plt.show()
