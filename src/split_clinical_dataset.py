import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "clinical")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "clinical_split")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

classes = ["cancer", "healthy"]

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)

for cls in classes:
    files = os.listdir(os.path.join(RAW_DIR, cls))
    random.shuffle(files)

    total = len(files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, split_files in splits.items():
        for f in split_files:
            src = os.path.join(RAW_DIR, cls, f)
            dst = os.path.join(SPLIT_DIR, split, cls, f)
            shutil.copy(src, dst)

print("✅ Clinical dataset split completed successfully")
