**Mango Leaf Disease Classification Using Lightwight CNN model with Saliency Map**

This repository contains a deep learning pipeline to classify mango images into different categories using a custom Convolutional Neural Network (CNN). The model is trained, validated, and tested on a structured dataset of mango images.

**Overview**

The goal of this project is to classify mango images into one of 8 classes. The project leverages TensorFlow and Keras for deep learning and includes data preprocessing, augmentation, and evaluation metrics such as accuracy, confusion matrix, and ROC curves.

**Dataset**

For this classification task, we have used the public dataset MangoLeafBD, which contains images of mango leaves categorized into different classes based on the leaf condition or species.

Structure: The dataset is organized into folders, where each folder corresponds to a specific mango class.
Splits:
Training: 80% of the data
Validation: 10% of the data
Testing: 10% of the data

**Model Architecture**

The model is a custom Convolutional Neural Network with the following layers:

1. Convolutional and Batch Normalization layers with activation (ReLU)
2. MaxPooling layers to reduce spatial dimensions
3. Fully Connected layers for classification
4. Dropout layers for regularization
5. Output layer with Softmax activation for multiclass classification

**Key Hyperparameters**

1. Input Shape: (227, 227, 3)
2. Optimizer: Stochastic Gradient Descent (SGD)
3. Learning Rate: 0.0001
4. Loss Function: Categorical Crossentropy
5. Batch Size: 12
6. Epochs: 100
