
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
MODEL_PATH = ROOT / "urbanleaf_model.pth"
CLASSES_PATH = ROOT / "classes.json"

# CPU training
device = torch.device("cpu")
print("Using device:", device)

# Image preprocessing
image_size = 224

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Load images from folders
train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=test_transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=0
)

# Create MobileNetV2 without downloading pretrained weights
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(train_dataset.classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {average_loss:.4f} | "
        f"Training accuracy: {train_accuracy:.2f}%"
    )

# Evaluate on held-out test images
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels.to(device)).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"\nTest accuracy: {test_accuracy:.2f}%")

# Save the trained model and class labels
torch.save(model.state_dict(), MODEL_PATH)

with open(CLASSES_PATH, "w", encoding="utf-8") as file:
    json.dump(train_dataset.classes, file, indent=4)

print("Model saved to:", MODEL_PATH)
print("Class labels saved to:", CLASSES_PATH)
