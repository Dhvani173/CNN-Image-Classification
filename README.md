

# CNN Image Classification Capstone Project
This is a project to demonstrate use of CNN to classify the images into two classes: cats and dogs.

# Project Overview
In this project, we train a CNN model to classify images of cats and dogs using a dataset of labeled images. The project includes data preprocessing, CNN model building, training, and evaluating the model on test data. The model is implemented using TensorFlow and Keras.

# Dataset
The dataset consists of images of cats and dogs. The images are organized into two main directories:

training_set: Contains labeled images for training (cats and dogs).

test_set: Contains labeled images for evaluating the model's performance.

Additionally, there's a single_prediction directory where users can input a new image to predict whether it's a cat or a dog.

The link to the dataset is: [/kaggle/input/ml-project-dataset](https://www.kaggle.com/datasets/dhvanibpatel173/ml-project-dataset)

# Imported libraries
Python 

TensorFlow

Keras

NumPy

Matplotlib

# Model Description
The model is a simple Convolutional Neural Network (CNN) consisting of:

•	Convolutional Layers: To extract features from images.

•	MaxPooling Layers: To reduce the dimensionality.

•	Fully Connected Layers: To classify the extracted features.

•	Dropout: To reduce overfitting.

•	Sigmoid Activation: For binary classification (Cat or Dog).

