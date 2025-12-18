import os
import shutil

SOURCE_DIR = r"C:\Users\Menaka\Downloads\H&E-stained oral squamous cell carcinoma histological images dataset\H&E-stained oral squamous cell carcinoma histological images dataset"
TARGET_DIR = r"data/histo_classification"

def create_dirs():
    os.makedirs(os.path.join(TARGET_DIR, "cancer"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "healthy"), exist_ok=True)

def copy_all_images(src_folder, dst_folder, prefix):
    """Recursively copy ALL images inside a folder."""
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                src = os.path.join(root, file)
                dst = os.path.join(dst_folder, f"{prefix}_{file}")
                shutil.copy(src, dst)

def copy_images():
    for split in ["training", "testing"]:
        split_path = os.path.join(SOURCE_DIR, split, "tumor", "patch", "640x640")

        if not os.path.exists(split_path):
            continue

        print(f"Processing split: {split}")

        for case in os.listdir(split_path):
            case_path = os.path.join(split_path, case)

            # ROI ⇒ cancer
            roi_original = os.path.join(case_path, "01-roi", "01-original")
            if os.path.exists(roi_original):
                print(f"  Copying ROI images → cancer ({case})")
                copy_all_images(roi_original, os.path.join(TARGET_DIR, "cancer"), f"{case}_roi")

            # NON-ROI ⇒ healthy
            non_roi = os.path.join(case_path, "02-non_roi")
            if os.path.exists(non_roi):
                print(f"  Copying NON-ROI images → healthy ({case})")
                copy_all_images(non_roi, os.path.join(TARGET_DIR, "healthy"), f"{case}_nonroi")

    print("\n✔ Dataset prepared successfully!")

if __name__ == "__main__":
    create_dirs()
    copy_images()
