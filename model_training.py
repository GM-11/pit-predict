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

# Oversample minority to 20% ratio for better balance
target_size = int(len(df_majority) / 2.5)
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
X_scaled = pd.DataFrame(scalar.fit_transform(X), columns=X.columns)

# Save the scaler BEFORE applying feature weights
import joblib

scalar = preprocessing.MinMaxScaler()
X_scaled = pd.DataFrame(scalar.fit_transform(X), columns=X.columns)
joblib.dump(scalar, "./model/scaler.pkl")

# Apply enhanced feature weights AFTER scaling and saving scaler
X = X_scaled.copy()
X["tyre_age"] *= 2.5  # Slightly reduced from 3
X["lap_number"] *= 2.0  # Increased importance
X["position"] *= 1.2
# First split into train+val and test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Then split train+val into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,  # 0.2 of 0.8 = 16% of total
)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_val = pd.DataFrame(X_val, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
y_train = pd.Series(y_train, name=y.name)
y_val = pd.Series(y_val, name=y.name)
y_test = pd.Series(y_test, name=y.name)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")
X_train


# %%cell 8
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)


# %%cell 9
class PitstopModel(torch.nn.Module):
    def __init__(self, input_size):
        super(PitstopModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


# %%cell 10
class_0_count = (y_train_tensor == 0).sum().float()
class_1_count = (y_train_tensor == 1).sum().float()
weight_ratio = class_0_count / class_1_count
pos_weight = torch.tensor([weight_ratio * 0.4], dtype=torch.float32).to(device)
print(f"Positive class weight: {pos_weight.item():.2f}")

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
model = PitstopModel(X_train_tensor.shape[1]).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.001, weight_decay=0.01
)  # Lower LR, weight decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=500, T_mult=2
)

# %%cell 11
epochs = 8000
train_losses = []
val_losses = []
learning_rates = []
best_val_loss = float("inf")
patience = 1000
patience_counter = 0
best_model_state = None

for epoch in range(epochs):
    # Training step
    model.train()
    y_pred = model(X_train_tensor)
    train_loss = criterion(y_pred.squeeze(), y_train_tensor)

    optimizer.zero_grad()
    train_loss.backward()

    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        val_loss = criterion(y_val_pred.squeeze(), y_val_tensor)

    # Step scheduler (use validation loss for plateau detection)
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()

    # Track metrics
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    current_lr = optimizer.param_groups[0]["lr"]
    learning_rates.append(current_lr)

    # Early stopping based on validation loss
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1

    # Print progress
    if (epoch + 1) % 200 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss.item():.4f}, "
            f"Val Loss: {val_loss.item():.4f}, LR: {current_lr:.6f}, "
            f"Patience: {patience_counter}/{patience}"
        )

    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        # Load best model
        # Check if we have a best model state before loading
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        else:
            print("Warning: No best model state found to load")
        break

# %%cell 12
# Plot 1: Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss", color="blue", alpha=0.7)
plt.plot(val_losses, label="Validation Loss", color="red", alpha=0.7)
# Add smoothed lines
train_smoothed = []
val_smoothed = []
if len(train_losses) > 100:
    train_smoothed = [
        sum(train_losses[max(0, i - 50) : i + 1]) / min(i + 1, 51)
        for i in range(len(train_losses))
    ]
    val_smoothed = [
        sum(val_losses[max(0, i - 50) : i + 1]) / min(i + 1, 51)
        for i in range(len(val_losses))
    ]
    plt.plot(train_smoothed, label="Smoothed Train", color="darkblue", linewidth=2)
    plt.plot(val_smoothed, label="Smoothed Val", color="darkred", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("images/training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 2: Learning Rate over time
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, label="Learning Rate", color="green")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale("log")  # Log scale for better visualization
plt.savefig("images/learning_rate.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 3: Train-Val Loss Difference (Overfitting indicator)
plt.figure(figsize=(10, 6))
loss_diff = []
if len(train_losses) > 100:
    loss_diff = [val - train for train, val in zip(train_losses, val_losses)]
    plt.plot(loss_diff, color="purple", alpha=0.7)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    plt.axhline(
        y=0.05, color="red", linestyle="--", alpha=0.5, label="Overfitting Warning"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss - Train Loss")
    plt.title("Overfitting Monitor")
    plt.legend()
    plt.grid(True, alpha=0.3)
plt.savefig("images/overfitting_monitor.png", dpi=300, bbox_inches="tight")
plt.show()

# Combined plot for overall reference
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Training Loss", color="blue", alpha=0.7)
plt.plot(val_losses, label="Validation Loss", color="red", alpha=0.7)
if len(train_losses) > 100:
    plt.plot(train_smoothed, label="Smoothed Train", color="darkblue", linewidth=2)
    plt.plot(val_smoothed, label="Smoothed Val", color="darkred", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(learning_rates, label="Learning Rate", color="green")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale("log")

plt.subplot(1, 3, 3)

if len(train_losses) > 100:
    plt.plot(loss_diff, color="purple", alpha=0.7)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    plt.axhline(
        y=0.05, color="red", linestyle="--", alpha=0.5, label="Overfitting Warning"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss - Train Loss")
    plt.title("Overfitting Monitor")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("images/training_progress_with_lr.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFinal Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
print(f"Final Train Loss: {train_losses[-1]:.4f}")
print(f"Final Val Loss: {val_losses[-1]:.4f}")
print(f"Best Val Loss: {best_val_loss:.4f}")
print(
    f"Best Train Loss: {min(train_losses):.4f} at epoch {train_losses.index(min(train_losses)) + 1}"
)

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

    # Test more refined thresholds focusing on precision-recall balance
    thresholds = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}

    # Also track best precision threshold
    best_precision = 0
    best_precision_threshold = 0.5
    best_precision_metrics = {}

    print("=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)
    print(
        f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'Pos_Pred':<8} {'PPV*Rec':<8}"
    )
    print("-" * 80)

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

        # Calculate precision-recall product for balanced metric
        pr_product = precision * recall

        # Print row
        print(
            f"{threshold:<10.1f} {accuracy * 100:<9.2f}% {precision:<10.3f} {recall:<7.3f} {f1:<7.3f} {pos_predictions:<8} {pr_product:<7.3f}"
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

        # Track best precision (with minimum recall requirement)
        if precision > best_precision and recall > 0.3:  # Minimum 30% recall
            best_precision = precision
            best_precision_threshold = threshold
            best_precision_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pos_pred": pos_predictions,
                "y_pred": y_pred_binary,
            }

    print("=" * 80)
    print(f"BEST F1 PERFORMANCE: Threshold = {best_threshold}")
    print(f"Accuracy: {best_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(
        f"Positive Predictions: {best_metrics['pos_pred']} / {np.sum(y_test_np)} actual"
    )
    print("=" * 80)

    if best_precision_metrics:
        print(f"BEST PRECISION PERFORMANCE: Threshold = {best_precision_threshold}")
        print(f"Accuracy: {best_precision_metrics['accuracy'] * 100:.2f}%")
        print(f"Precision: {best_precision_metrics['precision']:.4f}")
        print(f"Recall: {best_precision_metrics['recall']:.4f}")
        print(f"F1 Score: {best_precision_metrics['f1']:.4f}")
        print(
            f"Positive Predictions: {best_precision_metrics['pos_pred']} / {np.sum(y_test_np)} actual"
        )
        print("=" * 80)

    # Plot threshold analysis
    import matplotlib.pyplot as plt

    thresholds_list = [r["threshold"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]

    # Plot 1: Precision vs Recall vs F1
    plt.figure(figsize=(10, 6))
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
    plt.savefig("images/metrics_vs_threshold.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 2: Number of positive predictions
    plt.figure(figsize=(10, 6))
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
    plt.savefig("images/positive_predictions.png", dpi=300, bbox_inches="tight")
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

    # Save the trained model
    model_save_path = "model/pitstopmodel.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")

    # Also save training info
    training_info = {
        "model_state_dict": model.state_dict(),
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "best_precision_threshold": best_precision_threshold
        if best_precision_metrics
        else best_threshold,
        "best_precision": best_precision
        if best_precision_metrics
        else best_metrics["precision"],
        "input_size": X_train_tensor.shape[1],
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses),
        "early_stopped": len(train_losses) < epochs,
    }
    torch.save(training_info, "model/pitstopmodel_full.pth")
    print("✅ Full model info saved to model/pitstopmodel_full.pth")
