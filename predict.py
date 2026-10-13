from pathlib import Path
import json
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "urbanleaf_model.pth"
CLASSES_PATH = ROOT / "classes.json"

device = torch.device("cpu")

# Load classes
with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    classes = json.load(file)

# Create model
model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_pil_image(image):

    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = probs.argmax().item()

    pred_class = classes[pred_idx]

    confidence = probs[pred_idx].item() * 100

    prob_dict = {
        classes[i]: round(probs[i].item() * 100, 2)
        for i in range(len(classes))
    }

    return pred_class, confidence, prob_dict


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    prediction, confidence, probs = predict_pil_image(image)

    print("\n------ UrbanLeaf Prediction ------")
    print("Prediction:", prediction)
    print(f"Confidence: {confidence:.2f}%")

    print("\nClass Probabilities:")

    for cls, score in probs.items():
        print(f"{cls}: {score:.2f}%")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict.py image.jpg")

    else:
        predict_image(sys.argv[1])