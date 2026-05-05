# Team Members:
Dana Ibrahim dyi230000
Sumaiya Lana ssl240001
Sabiba Rudana srr220008

# Project Overview
This project implements a custom neural network autoencoder (built from scratch using NumPy) to perform anomaly detection for credit card fraud detection. The model is trained on normal transactions and detects fraud based on reconstruction error.

## Dataset
The dataset is hosted publicly on AWS S3 and is automatically downloaded at runtime.
- Source: AWS S3
- URL: https://detect-credit-card-fraud.s3.amazonaws.com/creditcard.csv

## Requirements
Install dependencies before running:
pip install numpy pandas scikit-learn matplotlib

## To Run:
python3 fraud_detection.py
