Image Classification

Project Overview

This project implements an image classification model using ResNet18 to classify images of cats and dogs. The dataset used is the Microsoft Cats and Dogs dataset (https://www.microsoft.com/en-us/download/details.aspx?id=54765). The model is trained using PyTorch with transfer learning, and inference is performed on test images.

Project Structure

The project is organized into multiple Python scripts:

configuratation.py - Defines configurations.

dataset.py - Handles dataset loading and transformations.

model.py - Initializes the ResNet18 model with transfer learning.

train.py - Implements the model training and validation loop.

showresult.py - Contains helper functions to visualize input images.

prediction.py - Initializes the model, trains it, and performs inference on test images.

Dependencies

Ensure you have the following libraries installed:

pip install torch torchvision numpy matplotlib

Dataset

Download the Microsoft Cats and Dogs dataset from here and extract it into a directory named data/ with subfolders train/ and val/.

Model Training

Initialize the model: The model is based on ResNet18 with pre-trained weights from ImageNet.

Feature Extraction: The convolutional layers are frozen, and only the final fully connected layer is trained.

Training Loop: The model is trained using CrossEntropyLoss and optimized using SGD with momentum.


Results

The training accuracy is printed for each epoch.

The predicted class is displayed on test images.

Future Improvements

Experiment with different architectures (ResNet50, EfficientNet, etc.).

Implement data augmentation techniques for better generalization.

Fine-tune more layers instead of freezing all convolutional layers.

Author

Developed by [Muhammad Taufique Popal]