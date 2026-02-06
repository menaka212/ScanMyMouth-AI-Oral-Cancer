# split_clinical.py
import os, shutil, random

BASE = "data/raw/clinical"
OUT = "data/clinical_split"
SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}

for cls in ["cancer", "healthy"]:
    files = os.listdir(os.path.join(BASE, cls))
    random.shuffle(files)

    n = len(files)
    t1 = int(n * SPLIT["train"])
    t2 = int(n * (SPLIT["train"] + SPLIT["val"]))

    splits = {
        "train": files[:t1],
        "val": files[t1:t2],
        "test": files[t2:]
    }

    for s, imgs in splits.items():
        dst = os.path.join(OUT, s, cls)
        os.makedirs(dst, exist_ok=True)
        for img in imgs:
            shutil.copy(
                os.path.join(BASE, cls, img),
                os.path.join(dst, img)
            )

print("✅ Clinical dataset split completed")
