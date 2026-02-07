# Brain Tumor Detection using Deep Learning

A deep learning-based system for automated detection of brain tumors from MRI scans.

This project implements MRI image classification (binary or multi-class) using Convolutional Neural Networks. It includes a deployed web application for real-time inference.

Live Demo: https://brain-tumor-detection-gw2r.onrender.com  
GitHub Repository: https://github.com/Gauravmangate27/Brain-Tumor-Detection

## Overview

The system uses a CNN model trained on brain MRI images to classify the presence and type of tumors. The model is integrated into a web application that accepts image uploads and returns predictions with confidence scores.

**Important Disclaimer**  
This application is intended strictly for educational and research purposes. It must not be used as a substitute for professional medical diagnosis or clinical decision-making.

## Features

- MRI-based brain tumor classification using Convolutional Neural Networks  
- Support for binary (tumor / no tumor) and multi-class classification  
- Web interface for image upload and real-time prediction  
- Flask-based backend (architecture compatible with FastAPI migration)  
- Image preprocessing (resizing and normalization)  
- Evaluation using standard classification metrics  

## Model Architecture

- Custom CNN implemented in TensorFlow and Keras  
- Designed to support transfer learning with architectures such as ResNet50 or EfficientNet  
- Scalable structure suitable for experimentation  

Optional future extension:  
- Tumor segmentation using U-Net or similar encoder-decoder architectures (not enabled in current version)

## Datasets

The model was trained using publicly available datasets commonly used in medical imaging research:

| Dataset                        | Classes                                  | Approximate Size | Source/Reference                          |
|--------------------------------|------------------------------------------|------------------|-------------------------------------------|
| Kaggle Brain MRI Dataset       | Glioma, Meningioma, Pituitary, Normal    | ~5,700 images    | Kaggle                                    |
| BraTS (2015–2021)              | Multi-modal 3D MRI with segmentation masks | Varies           | Multimodal Brain Tumor Segmentation Challenge |

Dataset files are not included in the repository due to size limitations. Users must download and organize the data in the correct directory structure before training.

## Reported Performance

| Task            | Dataset          | Model               | Accuracy | Notes                              |
|-----------------|------------------|---------------------|----------|------------------------------------|
| Classification  | Kaggle MRI       | Custom CNN          | ~96%     | Multi-class                        |
| Classification  | Kaggle MRI       | EfficientNet (fine-tuned) | ~98–99% | Reported in literature             |
| Segmentation    | BraTS 2018/2021  | U-Net baseline      | High Dice score | Not implemented in current demo    |

Detailed training curves, confusion matrices, and example predictions are included in the notebooks folder.
<img width="620" height="516" alt="image" src="https://github.com/user-attachments/assets/9b2c822f-3ee7-4276-946e-446d938d3e26" />



1. Clone the repository
```bash
git clone https://github.com/Gauravmangate27/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection

Install dependencies

Bashpip install -r requirements.txt
Running the Application
Bashpython app.py
Open a browser and navigate to:
http://127.0.0.1:5000/
Model Training (Optional)
To retrain or fine-tune the model:

Download and prepare the dataset in the expected folder structure
Open notebooks/training.ipynb
Execute the training cells
Save the resulting model as classifier.h5 in the model/ directory

Inference
Single-image prediction is available through the web interface.
The backend can be extended to provide REST API endpoints for integration into other systems.
Possible Extensions

Integrate transfer learning models (ResNet, EfficientNet, etc.)
Implement tumor segmentation using U-Net
Add batch processing support
Deploy on GPU-enabled cloud platforms
Migrate backend from Flask to FastAPI

References

Multiple studies reporting ~96% accuracy with CNN-based brain tumor classification on public MRI datasets
EfficientNet-based approaches achieving ~98–99% accuracy in recent literature
BraTS challenge benchmarks for segmentation tasks

License
MIT License
Contributions
Contributions are welcome. Please feel free to open issues or submit pull requests.
