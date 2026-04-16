# Brain Tumor Detection using CNN
This project uses a **Convolutional Neural Network (CNN)** with **MobileNetV2 transfer learning** to detect brain tumors from MRI images.
## Technologies Used
- Python
- TensorFlow / Keras
- CNN
- MobileNetV2
- ImageDataGenerator
## Features
- MRI image preprocessing
- Data augmentation
- Transfer learning
- Brain tumor classification
## 📊 Model Performance
- Training Accuracy: ~92%  
- Validation Accuracy: ~90%  
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
## Project Structure
brain-tumor-detection-cnn
├── src/
│ ├── train_model.py
│ ├── model.py
│ ├── dataset_loader.py
│ ├── app.py
├── models/
│ └── brain_tumor_model.h5
├── outputs/
│ └── accuracy.png
├── requirements.txt
└── README.md
## Installation
```bash
pip install -r requirements.txt
python train_model.py
Author
Raunak Ghosh
