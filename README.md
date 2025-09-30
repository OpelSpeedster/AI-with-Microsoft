
# Project 2: Breast Cancer Classification using CNN

**Objective**: To build a breast cancer classifier on an IDC dataset that can accurately classify a histology image as benign or malignant.

## Dataset :

Invasive Ductal Carcinoma (IDC) is the most common subtype of all breast cancers. To assign an aggressiveness grade to a whole mount sample, pathologists typically focus on the regions which contain the IDC. As a result, one of the common pre-processing steps for automatic aggressiveness grading is to delineate the exact regions of IDC inside of a whole mount slide. Use this IDC_regular dataset (the breast cancer histology image dataset) from Kaggle. This dataset holds 2,77,524 patches of size 50×50 extracted from 162 whole mount slide images of breast cancer specimens scanned at 40x. Of these, 1,98,738 test negative and 78,786 test positive with IDC. The dataset is available in public domain and you can download it here. You’ll need a minimum of 3.02GB of disk space for this.

## Classification (CancerNet)

This repository contains the Jupyter Notebook (Cancer5.ipynb) for a binary classification task
focused on identifying Invasive Ductal Carcinoma (IDC) in breast cancer histopathology images.
The project utilizes a custom Convolutional Neural Network (CNN), which we refer to as
**CancerNet** , to classify image patches into two categories: IDC-positive (Malignant) and
Non-IDC (Benign/Normal).

## 🚀 Project Overview

The core of this project is training a deep CNN model on microscopic image patches. We
perform extensive data preparation, implement a deep learning architecture, and evaluate
performance using standard classification metrics, including a detailed Confusion Matrix and
Classification Report.

## 📋 Setup and Dependencies

To run the Cancer5.ipynb notebook locally, you need a Python environment with Jupyter
installed.

### Dependencies

The following key libraries are required. They can be installed using pip:
pip install tensorflow opencv-python numpy pandas matplotlib seaborn
scikit-learn
# Note: The notebook includes steps to ensure NumPy compatibility.

### Data Requirement

The notebook requires the **Breast Histopathology Images** dataset.

1. **Download:** The notebook uses opendatasets to potentially download this data from
    Kaggle.

```
dataset_url = 'https://www.kaggle.com/datasets/paultimothymooney/breast-histopat
hology-images'
od.download(dataset_url)
```
```
# Set the base directory to where the data was downloaded
base_dir = './Breast-histopathology-images'
# The dataset needs to be extracted to a local directory.
```

Before installing the dataset, It will ask for Kaggle API Key (include Username and Key) to
get access to the dataset. Refer to the Kaggle API Instruction in the official website.

2. **Download and Extract the Dataset:**
   
```
image_paths = []
labels = []

# The data is organized in folders named by patient ID, and inside each,
# there are subfolders '0' and '1' for the classes.
for patient_id in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient_id)
    if os.path.isdir(patient_path):
        for class_label in ['0', '1']:
            class_path = os.path.join(patient_path, class_label)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_paths.append(os.path.join(class_path, image_name))
                    labels.append(int(class_label))
```

3. **Structure:** Ensure your base data directory (./Breast-histopathology-images) is set
    correctly, with patient folders containing subfolders 0 (Non-IDC) and 1 (IDC).

## 🧠 Model Architecture (CancerNet)

The CancerNet model is a sequential CNN designed for robust feature extraction from small 50×50×3 image patches.

```
CancerNet = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

CancerNet.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

CancerNet.summary()
```

**Total Parameters:** 1,143,745 (1,143,297 Trainable)

## 🛠 Key Steps in the Notebook

1. **Environment Setup:** Installs and verifies necessary libraries, including handling NumPy
    version conflicts.
2. **Data Loading & Balancing:** Reads image paths and labels. A crucial step involves
    balancing the dataset by sampling 25,000 instances of each class to mitigate initial class
    imbalance.
3. **Data Splitting:** Splits the balanced data into Training (72%), Validation (8%), and Testing
    (20%) sets using stratified sampling.
4. **Data Augmentation:** Implements ImageDataGenerator for the training set, applying
    transformations (rotation, shifting, zoom, etc.) to enhance data diversity.
5. **Model Definition & Training:** Defines CancerNet and trains it for **25 epochs** using the
    Adam optimizer and binary_crossentropy loss.
6. **Evaluation:** Generates a Confusion Matrix, plots accuracy/loss over epochs, and prints a
    Classification Report on the held-out test set.


**Author:** S.V.Vishal


