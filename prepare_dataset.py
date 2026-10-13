from pathlib import Path
import random
import shutil

random.seed(42)

root = Path(__file__).resolve().parent
source = root / "plantvillage_temp" / "raw" / "color"
destination = root / "dataset"

classes = {
    "Tomato___healthy": "healthy",
    "Tomato___Early_blight": "diseased",
}

for folder in ["train/healthy", "train/diseased", "test/healthy", "test/diseased"]:
    target = destination / folder
    target.mkdir(parents=True, exist_ok=True)
    for old_file in target.iterdir():
        if old_file.is_file():
            old_file.unlink()

for source_name, target_name in classes.items():
    files = list((source / source_name).glob("*"))
    files = [f for f in files if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.shuffle(files)

    selected = files[:1000]
    if len(selected) < 1000:
        raise RuntimeError(f"Expected 1000 images in {source_name}, found {len(selected)}")

    train_files = selected[:800]
    test_files = selected[800:1000]

    for split, items in [("train", train_files), ("test", test_files)]:
        target = destination / split / target_name
        for index, file in enumerate(items):
            shutil.copy2(file, target / f"{source_name}_{index:04d}{file.suffix.lower()}")

    print(f"{target_name}: {len(train_files)} training, {len(test_files)} testing")

print("\nDataset preparation complete!")
