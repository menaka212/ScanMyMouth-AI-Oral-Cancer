# ScanMyMouth-AI-Oral-Cancer

An AI-powered oral cancer screening system that analyzes oral images, predicts cancer risk, and provides explainable results using Grad-CAM heatmaps.
Oral Cancer Detection using EfficientNet + Grad-CAM + ViT

This project detects oral cancer from medical images using:
- EfficientNetB0 (classification)
- Vision Transformer ViT (comparison model)
- Grad-CAM explainability
- Django web application (user interface)

## Folder Structure
- data/ → dataset (train, val, test)
- models/ → trained model files
- src/ → ML training scripts
- webapp/ → Django web app
- notebooks/ → training notebooks
- reports/ → final report & PPT

## Features
- AI-based Oral Cancer Detection
- Multi-class classification support
- Heatmap visualization
- Model comparison (EfficientNet vs ViT)
- Full Web App Integration


## 📊 Output
The system displays:
- Uploaded image
  <img width="1876" height="870" alt="Screenshot 2026-02-09 114004" src="https://github.com/user-attachments/assets/e1e725b4-1bf9-4c21-b483-9ce3244133b8" />

- AI prediction result
- Confidence score
- Risk level
- Grad-CAM heatmap highlighting lesion regions

## ⚠️ Disclaimer
This system is intended for **early screening and decision support only**.  
It is **not a replacement for professional medical diagnosis**.

## 👩‍💻 Developed By
Menaka Manavalan

