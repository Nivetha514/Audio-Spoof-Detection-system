# AI_ML_Spoof_detection_Real_vs_recorded_vs_generated_audio_detection

Audio Spoof Detection using Ensemble Architecture

## Overview
This project implements an ensemble architecture for audio spoof detection using the machine learning (ML) algorithms and deep learning models. The goal is to detect spoofed audio samples from genuine ones. Two separate datasets were used for training and evaluating the ensemble model, and its performance was assessed using accuracy, F1 score, recall, and precision metrics.

## Architecture
There are 2 ensemble architecture. Those are:
1. *Machine Learning Model*: The developed ensemble model combines various machine learning classifiers, including Random Forest, HistGradientBoosting, SVC, KNeighbors, and Decision Tree, to enhance predictive accuracy for audio-based spoofing detection. Utilizing two distinct datasets, "DATASET-balanced.csv" and "DATASET-LA.csv," the model ensures a balanced representation of classes through stratified sampling. The ensemble employs a soft voting strategy to aggregate individual model predictions, achieving robust performance. The model's effectiveness is demonstrated through comprehensive evaluations, encompassing accuracy, precision, and recall metrics, showcasing its versatility and high accuracy in different scenarios. This ensemble model offers a flexible and powerful solution for audio-based classification tasks.
2. *Deep Learning Model*: The proposed model is an ensemble learning system that combines three distinct neural network architectures—Convolutional Neural Network (CNN), Time-delayed Neural Network (TDNN), and Recurrent Neural Network (RNN). These architectures are designed to process time-series audio features extracted from the ASV Spoof Dataset 2021-LA and DEEP_VOICE dataset file. The ensemble leverages the strengths of each individual model, enhancing the overall predictive performance. The models are trained using a balanced dataset, utilizing binary cross-entropy loss and the Adam optimizer. The training process involves 10 epochs with a batch size of 32. The final ensemble predictions are obtained by averaging the output probabilities and converting them to binary decisions. The accuracy, f1-score, Precison, Recall of the ensemble is assessed and reported using scikit-learn
The predictions from these models are combined using a voting classifier to make the final decision.

## Datasets
Two datasets were used in this project:
1. ASV Spoof 2021 Dataset-LA: 
2. DEEP_VOICE Dataset: This dataset is comprised of real human speech from eight well-known figures and their speech converted to one another using Retrieval-based Voice Conversion. For each speech, the accompaniment ("background noise") was removed before conversion using RVC. The original accompaniment is then added back to the DeepFake speech

## Evaluation
The performance of the ensemble model was evaluated using the following metrics:
- Accuracy
- F1 Score
- Recall
- Precision

## Usage
To use this project, follow these steps:
1. *Install Dependencies*: Python: Make sure you have Python installed on your system. You can download it from https://www.python.org/.

Pandas: A powerful data manipulation library. Install it using:
pip install pandas


NumPy: A fundamental package for scientific computing with Python. Install it using:
pip install numpy


Scikit-learn: A machine learning library for simple and efficient tools. Install it using:
pip install scikit-learn


TensorFlow: An open-source machine learning framework. Install it using:
pip install tensorflow


Keras: A high-level neural networks API, running on top of TensorFlow. Install it using:
pip install keras


Matplotlib: A 2D plotting library. Install it using:
pip install matplotlib


Seaborn: A statistical data visualization library. Install it using:
pip install seaborn

Ensure that you have the appropriate versions of these dependencies to avoid compatibility issues. You can usually find the version requirements in the project's documentation or README file. Additionally, please note that GPU support might require specific configurations and installations.

2. *Training*: 
*DeepLearning Ensemble:*
Data Preparation:
Load your datasets using Pandas. In the given example, two CSV files, file1.csv and file2.csv, are loaded and concatenated into a single dataset.
Extract features (X) and labels (y) from the dataset. Encode labels using LabelEncoder.
Train-Test Split:
Split the data into training and testing sets using train_test_split.
Reshape Data:
Reshape the input data for TDNN assuming the time dimension is the second axis.
Model Definition:
Define three different models (TDNN, RNN, CNN) using Keras. In the provided code, Sequential models are used.
Model Compilation:
Compile each model with the specified loss function, optimizer, and metrics. In this case, binary cross-entropy is used as the loss function, 'adam' as the optimizer, and accuracy as the metric.
Model Training:
Train each model using the training data and validate on the testing data. In the provided code, each model is trained for 20 epochs.
Ensemble Predictions:
Make predictions using each model on the test set. Combine these predictions by averaging and apply a threshold to get the ensemble predictions.
Evaluate the Ensemble Model:
Evaluate the ensemble model on the test set. Convert predictions back to the original labels using label_encoder.inverse_transform and calculate accuracy.
Print Ensemble Accuracy:
Print the accuracy of the ensemble model on the test set.


*Machine Learning Ensemble:*
Data Preparation:
Utilize diverse datasets, e.g., Random Forest and Gradient Boosting, ensuring balance through stratified sampling. Train models and encode labels for a comprehensive approach.
Model Selection:
Choose classifiers like Random Forest and Gradient Boosting for diversity, contributing to a robust ensemble. This ensures varied perspectives and enhances predictive power.
Training Process:
Train selected models on your datasets, adjusting hyperparameters for optimal performance. Utilize stratified sampling to maintain balance in the data distribution.
Label Encoding:
Encode labels using appropriate techniques, ensuring compatibility with the ensemble training process.
Voting Classifier Creation:
Create a voting classifier to combine predictions from diverse models, promoting ensemble learning and capturing multiple facets of the data.
Evaluation Metrics:
Assess the ensemble's performance using metrics like accuracy, precision, and recall. This provides a comprehensive understanding of predictive efficacy.
Dependencies:
Ensure the presence of essential dependencies, including Python, scikit-learn, and pandas, for seamless execution.
Pre-trained Models (Optional):
If using pre-trained models, load them into the ensemble, following the same process. Adapt the code for specific datasets and classifiers.
Optimization Strategies:
Experiment with hyperparameter tuning and model combinations to achieve optimal results. Adjust the ensemble composition based on the characteristics of your data.


## Contributors
- Soundarya Lahari: Developer, Researcher
- Nivetha S: Developer, Researcher

