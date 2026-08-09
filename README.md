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
  <img width="1895" height="863" alt="Screenshot 2026-02-09 114725" src="https://github.com/user-attachments/assets/70a51993-3998-479d-a890-89d5ade34dd1" />
  <img width="1895" height="863" alt="Screenshot 2026-02-09 115122" src="https://github.com/user-attachments/assets/a8f6915f-77e6-44b5-ab0e-b6e3c51bf457" />
  <img width="1895" height="863" alt="Screenshot 2026-02-09 115637" src="https://github.com/user-attachments/assets/61e793d1-3347-4036-9eb6-c160d2bd9393" />

  
- Confidence Score and Risk Level
  <img width="1862" height="868" alt="Screenshot 2026-02-09 115204" src="https://github.com/user-attachments/assets/e97534e3-319a-478c-9a2a-8a0279937ab3" />
  <img width="1882" height="860" alt="Screenshot 2026-02-09 120148" src="https://github.com/user-attachments/assets/b0415911-f1ed-45a1-b43e-8f0b17d5113d" />

- Grad-CAM heatmap highlighting lesion regions
  <img width="1873" height="857" alt="Screenshot 2026-02-09 115131" src="https://github.com/user-attachments/assets/33198522-34fe-42dd-a00a-45d36932fbcb" />
  <img width="1871" height="865" alt="Screenshot 2026-02-09 115648" src="https://github.com/user-attachments/assets/ac8def49-3565-4b04-9e2d-0fa105f38ddf" />
    
## ⚠️ Disclaimer
This system is intended for **early screening and decision support only**.  
It is **not a replacement for professional medical diagnosis**.

## 👩‍💻 Developed By
Menaka Manavalan

