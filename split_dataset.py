import os
import shutil
import random

# Paths
SOURCE_DIR = "data/raw"   # where your original folders are
TARGET_DIR = "data"       # output split folders

# Split ratios
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

def create_dirs(classes):
    """Create train/val/test directories with class subfolders."""
    for split in SPLIT_RATIOS.keys():
        for cls in classes:
            path = os.path.join(TARGET_DIR, split, cls)
            os.makedirs(path, exist_ok=True)

def split_dataset():
    """Split dataset into train, val, test automatically."""
    classes = [cls for cls in os.listdir(SOURCE_DIR) 
               if os.path.isdir(os.path.join(SOURCE_DIR, cls))]

    create_dirs(classes)

    for cls in classes:
        class_path = os.path.join(SOURCE_DIR, cls)
        images = [img for img in os.listdir(class_path)
                  if img.lower().endswith((".jpg", ".jpeg", ".png"))]

        random.shuffle(images)

        total = len(images)
        train_end = int(total * SPLIT_RATIOS["train"])
        val_end = train_end + int(total * SPLIT_RATIOS["val"])

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        print(f"\nClass: {cls}")
        print(f"Total: {total}")
        print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

        # Copy files
        for img, group in zip([train_imgs, val_imgs, test_imgs], SPLIT_RATIOS.keys()):
            for file in img:
                src = os.path.join(class_path, file)
                dst = os.path.join(TARGET_DIR, group, cls, file)
                shutil.copy(src, dst)

    print("\n✔ Dataset split completed successfully!")

if __name__ == "__main__":
    split_dataset()
