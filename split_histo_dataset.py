import os
import shutil
import random

SOURCE_DIR = "data/histo_classification"
TARGET_DIR = "data/histo_split"

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

def create_dirs(classes):
    for split in SPLIT_RATIOS.keys():
        for cls in classes:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

def split_dataset():
    classes = os.listdir(SOURCE_DIR)
    create_dirs(classes)

    for cls in classes:
        class_path = os.path.join(SOURCE_DIR, cls)
        images = [img for img in os.listdir(class_path)
                  if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]

        random.shuffle(images)

        total = len(images)
        train_end = int(total * SPLIT_RATIOS["train"])
        val_end = train_end + int(total * SPLIT_RATIOS["val"])

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        for img_set, split in zip([train_imgs, val_imgs, test_imgs], SPLIT_RATIOS.keys()):
            for img in img_set:
                src = os.path.join(class_path, img)
                dst = os.path.join(TARGET_DIR, split, cls, img)
                shutil.copy(src, dst)

    print("✔ Histo dataset split successfully!")

if __name__ == "__main__":
    split_dataset()
