import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time

# ================== Reproducibility ==================
seed = 22
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================== Paths ==================
train_dir = "training_aug"
test_dir = "testing"
os.makedirs("checkpoints", exist_ok=True)
epochs = 30

# ================== Device ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== Transforms ==================
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=data_transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)

# ================== Model ==================
# Use ResNet18
model = models.resnet18(pretrained=True)

# Replace the final fully connected layer
num_ftrs = model.fc.in_features



model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)
#model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device)


# ================== Loss & Optimizer ==================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

# ================== Training Loop ==================
train_acc_history = []
val_acc_history = []
best_acc = 0.0
best_epoch = 0

print("\n⏳ Starting training...")
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()

    model.train()
    running_corrects = 0
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()

    train_acc = running_corrects / len(train_loader.dataset)
    train_acc_history.append(train_acc)

    # ===== Validation =====
    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            val_corrects += (preds == labels).sum().item()
    val_acc = val_corrects / len(test_loader.dataset)
    val_acc_history.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "checkpoints/resnet18_state_dict.ckpt")

    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f} - Time: {epoch_time:.2f}s")

total_time = time.time() - start_time
print(f"\n✅ Best Val Accuracy: {best_acc:.4f} at epoch {best_epoch}")
print(f"⏱️ Total Training Time: {total_time/60:.2f} minutes")

# ================== Plot Accuracy ==================
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_acc_history, label="Train Acc")
plt.plot(range(1, epochs+1), val_acc_history, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("SqueezeNet Training Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ================== Testing (Accuracy, Precision, Recall, F1) ==================
print("\n🔎 Loading best model for testing...")
model.load_state_dict(torch.load("checkpoints/resnet18_state_dict.ckpt"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ====== Metrics (macro instead of weighted) ======
test_acc  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print("\n📊 Test Results:")
print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ===== Detailed Report =====
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes, zero_division=0))
