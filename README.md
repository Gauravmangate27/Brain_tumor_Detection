Brain Tumor Detection using Deep Learning
A deep learning–based system for automated brain tumor detection from MRI scans.
The project focuses on MRI image classification (binary or multi-class) using Convolutional Neural Networks (CNNs), with optional support for future tumor segmentation.
Live Demo: https://brain-tumor-detection-gw2r.onrender.com
GitHub Repository: https://github.com/Gauravmangate27/Brain-Tumor-Detection
________________________________________
Overview
This project demonstrates the application of deep learning techniques in medical image analysis. A CNN-based model is trained on brain MRI images to identify the presence and type of brain tumors. The trained model is deployed as a web application that allows users to upload MRI images and receive real-time predictions with confidence scores.
Disclaimer:
This application is intended for educational and research purposes only and must not be used as a substitute for professional medical diagnosis.
________________________________________
Features
•	MRI-based brain tumor classification using Convolutional Neural Networks
•	Support for binary (tumor / no tumor) or multi-class classification
•	Web-based interface for image upload and real-time prediction
•	REST-style backend implemented using Flask (FastAPI-ready architecture)
•	Image preprocessing including resizing and normalization
•	Model evaluation using standard classification metrics
________________________________________
Model Architecture
•	Custom CNN built using TensorFlow and Keras
•	Extendable to transfer learning backbones such as ResNet50 or EfficientNet
•	Designed for scalability and experimentation
Optional extension:
•	Tumor segmentation using U-Net or similar architectures (not enabled by default)
________________________________________
Dataset
The model is trained using publicly available brain MRI datasets commonly used in academic research:
Common Datasets
•	Kaggle Brain MRI Dataset
o	Classes: Glioma, Meningioma, Pituitary, Normal
o	Approx. 5,700 MRI images
•	BraTS (2015–2021) (for segmentation research)
o	Multi-modal 3D MRI scans with pixel-level tumor masks
Dataset files are not included in this repository due to size constraints.
Ensure correct dataset structure, version, and preprocessing before training.
________________________________________
Performance Metrics
The system supports evaluation using the following metrics:
•	Accuracy
•	Precision
•	Recall
•	F1-score
•	ROC-AUC
•	Confusion Matrix
Sample Results
Task	Dataset	Performance
Classification	Kaggle MRI	~96% accuracy
Segmentation (baseline)	BraTS 2018/2021	High Dice score (U-Net baseline)
Training curves, confusion matrices, and inference examples are available in the notebooks.
________________________________________
Project Structure
Brain-Tumor-Detection/
│
├── app.py                 # Flask web application
├── model/
│   └── classifier.h5      # Trained CNN model
├── static/
│   └── uploads/           # Uploaded MRI images
├── templates/
│   └── index.html         # Frontend UI
├── notebooks/
│   └── training.ipynb     # Model training & evaluation
├── requirements.txt
└── README.md
________________________________________
Installation & Setup
Clone the Repository
git clone https://github.com/Gauravmangate27/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
Install Dependencies
pip install -r requirements.txt
________________________________________
Running the Application
python app.py
Open your browser and navigate to:
http://127.0.0.1:5000/
________________________________________
Model Training (Optional)
To retrain the classification model:
1.	Prepare the dataset with appropriate directory structure
2.	Open the training notebook in notebooks/
3.	Train the CNN model
4.	Save the trained model as .h5 and place it in the model/ directory
________________________________________
Inference
Single-image prediction is supported via the web interface.
The backend can be easily extended to expose REST APIs for external integration.
________________________________________
Extending the Project
•	Integrate transfer learning models (ResNet, EfficientNet)
•	Add tumor segmentation using U-Net
•	Enable batch inference
•	Deploy on GPU-enabled cloud platforms
•	Convert Flask backend to FastAPI for scalable inference
________________________________________
References
•	CNN-based brain tumor classification achieving ~96% accuracy (multiple studies)
•	EfficientNet-based multi-class MRI classification (~99% reported in literature)
•	U-Net segmentation performance in BraTS 2021 benchmarks
________________________________________
License
This project is licensed under the MIT License.
________________________________________
Contributions
Contributions are welcome.
Feel free to fork the repository, open issues, or submit pull requests.
________________________________________
Why This Project Structure Works
•	Clear problem definition and scope
•	Reproducible setup and usage instructions
•	Metrics aligned with academic and industry standards
•	Designed for extensibility and future research

