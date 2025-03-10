# Digit Classifier

A simple web application that lets you draw digits and see how different machine learning models recognize them. Built as a learning project to explore image recognition techniques.

## About

This app demonstrates machine learning concepts by comparing:
- K-Nearest Neighbors algorithm (traditional ML)
- Neural Network approach (deep learning)
- ...work in progress

All models are trained on the MNIST dataset.

## Features

- Drawing canvas where you can sketch digits
- Side-by-side comparison of model predictions
- Visualization of the image processing steps
- Super simple responsive interface

## Tech Stack

- Python with Dash for the web interface
- TensorFlow/Keras for the neural network model
- Scikit-learn for the KNN model
- Plotly for visualizations

## How to Use
1. Clone the repository
```bash

git clone https://github.com/ludwilton/digit-classifier.git
cd digit-classifier
```
2. install dependencies
```bash
pip install -r requirements.txt
```
3. run the app
```bash
python main.py
```
4. Open your browser to http://127.0.0.1:8050/
5. Draw a digit!

## Project structure

main.py - The application entry point

layout.py - UI components

data_util.py - Image processing and prediction functions

neural_network.ipynb - Training notebook