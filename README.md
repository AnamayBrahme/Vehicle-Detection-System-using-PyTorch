# Vehicle Detection using Faster R-CNN

This project implements a vehicle detection system using the Faster R-CNN model, trained on a custom dataset of vehicle images with bounding box annotations.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Future Improvements](#future-improvements)

## Project Overview

This project aims to detect vehicles in images using a Faster R-CNN model. The dataset consists of images with bounding box annotations for various vehicle types, including cars, buses, motorcycles, pickups, and trucks. The project includes data preprocessing, model training, evaluation, and visualization of the results.

## Dataset

The dataset is stored in JSON format, with separate files for training and testing data. It contains information about images, annotations (bounding boxes and category IDs), and categories. The dataset is organized as follows:

-   `train_img_path`: Path to the training images.
-   `train_label_file`: Path to the training labels in JSON format.
-   `test_img_path`: Path to the testing images.
-   `test_label_file`: Path to the testing labels in JSON format.

The dataset includes the following categories:

-   cars
-   Bus
-   Car
-   Motorcycle
-   Pickup
-   Truck

## Dependencies

-   Python 3.x
-   PyTorch
-   Torchvision
-   NumPy
-   PIL (Pillow)
-   OpenCV (cv2)
-   Matplotlib
-   Plotly

You can install the required dependencies using pip:

```bash
pip install torch torchvision numpy Pillow opencv-python matplotlib plotly
