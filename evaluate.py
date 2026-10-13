
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset" / "test"
MODEL_PATH = ROOT / "urbanleaf_model.pth"
OUTPUT_PATH = ROOT / "confusion_matrix.png"

device = torch.device("cpu")

# Use the same preprocessing as training and prediction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Load the held-out test dataset
test_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

class_names = test_dataset.classes
print("Class order:", class_names)
print("Test images:", len(test_dataset))

# Rebuild the same MobileNetV2 architecture
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names),
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.to(device)
model.eval()

# Predict every test image
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())

# Calculate evaluation metrics
accuracy = accuracy_score(all_labels, all_predictions)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    zero_division=0,
)

print("\n--- UrbanLeaf Evaluation Results ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

for index, name in enumerate(class_names):
    print(f"\nClass: {name}")
    print(f"Precision: {precision[index] * 100:.2f}%")
    print(f"Recall:    {recall[index] * 100:.2f}%")
    print(f"F1-score:  {f1[index] * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
)

# Create and save the confusion matrix
matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
)

print("Confusion matrix:")
print(matrix)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names,
)
display.plot(cmap="Blues", values_format="d")
plt.title("UrbanLeaf - Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
plt.close()

print(f"\nConfusion matrix saved to: {OUTPUT_PATH}")
