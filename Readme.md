# Project Name: Predicting Wound Locations in Animal Images

## Introduction

This project uses machine learning techniques to predict wound locations in animal images. It involves multiple models, including **Linear Regression**, **Ridge Regression**, and **Random Forest**, to predict the wound's coordinates (x, y, width, height) based on image data. The project is implemented using Jupyter notebooks and Python scripts.

## Project Structure

The project consists of the following main components:

1.  **Utils.py** – Contains common utility functions used across the notebooks and scripts.
2.  **Trainer-LinearRegression.py** – A script to train Linear Regression and Ridge Regression models.
3.  **Trainer-RandomForest.py** – A script to train Random Forest models.
4.  **Tester-LinearRegression.py** – A script to test the trained Linear Regression and Ridge Regression models on a test dataset and visualize the results.
5.  **Tester-RandomForest.py** – A script to test the trained Random Forest models on a test dataset and visualize the results.

## Prerequisites

-   Python 3.x
-   Jupyter Notebooks
-   Required libraries:
    -   NumPy
    -   Pandas
    -   Matplotlib
    -   scikit-learn
    -   Joblib
    -  os
    -  PIL
    -  csv

You can install the required libraries using the following command:

`pip install -r requirements.txt` 

## Files Overview

### 1. **Utils.py**

This file contains utility functions that are used in training and testing models:

-   `readImageData(rootpath)` – Reads the image data from the provided root path.
-   `display_images(images, rows, cols, titles=None)` – Displays images in a grid.
-   `calculate_error(pred, gt)` – Computes the error between predictions and ground truth.
-   `draw_rectangle(ax, center_x, center_y, x_width, y_width, color, scale_factor_x, scale_factor_y)` – Draws a rectangle on the image for visualizing wound locations.

### 2. **Trainer-LinearRegression.py**

This file trains **Linear Regression** and **Ridge Regression** models. The models are saved in the `linear_regression_models` folder.

To run the trainer, execute the following:

`python Trainer-LinearRegression.py` 

### 3. **Trainer-RandomForest.py**

This file trains **Random Forest** models. The models are saved in the `random_forest_models` folder.

To run the trainer, execute the following:


`python Trainer-RandomForest.py` 

### 4. **Tester-LinearRegression.py**

This file loads the trained Linear Regression and Ridge Regression models from the `linear_regression_models` folder, tests them on the test dataset, and visualizes the predictions.

To run the tester, execute the following:

`python Tester-LinearRegression.py` 

### 5. **Tester-RandomForest.py**

This file loads the trained Random Forest models from the `random_forest_models` folder, tests them on the test dataset, and visualizes the predictions.

To run the tester, execute the following:

`python Tester-RandomForest.py` 

## How to Run the Code

1.  **Train Models**:
    
    -   Start by training the models using the respective trainer scripts (`Trainer-LinearRegression.py` or `Trainer-RandomForest.py`). The trained models will be saved in the `linear_regression_models` or `random_forest_models` folders.
2.  **Test Models**:
    
    -   Once the models are trained, you can test them by running the `Tester-LinearRegression.py` or `Tester-RandomForest.py` scripts. These will load the models and visualize the prediction results for the test dataset.

## Visualizations

The testers will generate visualizations that show the predicted wound location (in red) and the actual ground truth (in green) on the images. The visualizations will help you evaluate the performance of the models in predicting the wound locations. And the visualized results is saved in folder `results`.

## Future Work

-   Experiment with other advanced models such as deep learning techniques.
-   Optimize the feature extraction and data preprocessing pipeline to improve model accuracy.
-   Investigate further the effects of dimensionality reduction techniques like PCA on model performance.
