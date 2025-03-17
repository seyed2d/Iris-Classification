# Iris Classification with PyTorch

This project implements a machine learning model for classifying iris flowers using the popular Iris dataset. The classification model is built using PyTorch and evaluated with K-Fold Cross Validation.

## Project Overview

The goal of this project is to classify iris flowers into three species: Setosa, Versicolor, and Virginica. The dataset contains four features for each flower: sepal length, sepal width, petal length, and petal width.

A deep learning model is built using PyTorch with the following architecture:
- 4 input features (sepal length, sepal width, petal length, petal width)
- 1 hidden layer with 16 neurons
- 2 hidden layers with 16 and 8 neurons
- 1 output layer with 3 neurons corresponding to the three classes

K-Fold Cross Validation is used to evaluate the model, training the model on different subsets of the dataset and evaluating its accuracy.

## Dataset

The dataset used is the famous Iris Dataset which consists of 150 instances with 4 features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The target variable consists of three species of iris flowers: Setosa, Versicolor, and Virginica.

## Model

The model architecture consists of the following layers:
1. Input Layer: 4 input features (corresponding to the 4 attributes of the iris flowers).
2. Hidden Layer 1: 16 neurons.
3. Hidden Layer 2: 16 neurons.
4. Hidden Layer 3: 8 neurons.
5. Output Layer: 3 neurons, corresponding to the 3 classes (Setosa, Versicolor, and Virginica).

The model uses the CrossEntropyLoss criterion for training and the Adam optimizer for weight updates.

## Training and Evaluation

The model is trained using K-Fold Cross Validation with the following steps:

1. Split the dataset into 5 folds using K-Fold.
2. For each fold, the model is trained on 80% of the data and tested on the remaining 20%.
3. The accuracy is calculated for both the training and testing datasets.
4. The training and testing accuracy for each fold are printed.
