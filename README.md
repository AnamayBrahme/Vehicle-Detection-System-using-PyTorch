# Vehicle Detection using Faster R-CNN

## Overview
This project implements a vehicle detection system using the Faster R-CNN model. The model is trained to identify various types of vehicles in images, including cars, buses, motorcycles, pickups, and trucks. The dataset used for training and testing is structured in a JSON format, containing annotations for bounding boxes and categories.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Main Processes and Components](#main-processes-and-components)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Results](#results)
- [License](#license)

## Installation
To run this project, you need to have Python 3.x installed along with the following libraries:

```bash
pip install torch torchvision numpy matplotlib opencv-python pillow
```

## Dataset
The dataset used in this project is structured in a JSON format with the following keys:
Dataset - https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset/data

- `images`: Metadata for each image (ID, filename, dimensions).
- `annotations`: Bounding box information for each object in the images.
- `categories`: Information about the classes of objects.

### Dataset Structure
```
/path/to/dataset/
    ├── train/
    │   ├── images/
    │   ├── annotations.json
    └── test/
        ├── images/
        ├── annotations.json
```

## Usage
Clone the repository:

```bash
git clone https://github.com/AnamayBrahme/vehicle-detection.git
cd vehicle-detection
```

Update the paths in the code to point to your dataset.

Run the training script:

```bash
python train.py
```

## Main Processes and Components

### Overview
The vehicle detection project utilizes the Faster R-CNN (Region-based Convolutional Neural Network) architecture to identify and classify various types of vehicles in images. The model is trained on a custom dataset containing images and their corresponding annotations in JSON format.

### Key Components

1. **Data Preparation**
   - **Dataset Structure**: The dataset is organized into training and testing sets, each containing images and a JSON file with annotations. The annotations include bounding box coordinates, category IDs, and other metadata.
   - **Data Loading**: The `VehicleDetectionDataset` class is implemented to load images and their corresponding annotations. It processes the data into a format suitable for training the model.

2. **Data Augmentation**
   - **Transformations**: To improve the model's robustness, data augmentation techniques are applied. This includes random horizontal flips of images and their bounding boxes. The `Compose` class manages a series of transformations to be applied to the images and bounding boxes.

3. **Model Architecture**
   - **Faster R-CNN**: The project uses the Faster R-CNN model with a ResNet-50 backbone. This architecture is well-suited for object detection tasks, as it combines region proposal networks with a fast R-CNN detector.
   - **Custom Head**: The model's head is modified to match the number of classes in the dataset. The `FastRCNNPredictor` is used to replace the default head of the Faster R-CNN model.

4. **Training Process**
   - **Loss Calculation**: During training, the model computes the loss for both classification and bounding box regression. The training loop iterates over the dataset, updating the model weights based on the computed losses.
   - **Learning Rate Scheduling**: A learning rate scheduler is used to adjust the learning rate during training, which helps in converging to a better solution.

5. **Evaluation**
   - **Metrics Calculation**: After training, the model is evaluated on the test dataset. Metrics such as Intersection over Union (IoU) are calculated to assess the model's performance in detecting and classifying vehicles.
   - **Visualization**: The project includes functions to visualize the model's predictions on test images. This helps in understanding how well the model performs and where it may be making errors.

6. **Results Visualization**
   - **Class Frequency Histograms**: The project generates histograms to visualize the frequency of each class in the training and testing datasets. This provides insights into the distribution of vehicle types in the dataset.

### Conclusion
This project demonstrates the application of deep learning techniques for vehicle detection using the Faster R-CNN architecture. By leveraging data augmentation, a robust model architecture, and effective training strategies, the system aims to achieve high accuracy in detecting various types of vehicles in images.

## Evaluation
After training, the model can be evaluated on the test dataset. The evaluation includes calculating metrics such as Intersection over Union (IoU) and visualizing the predictions.

## Visualization
The project includes functions to visualize the images along with their predicted bounding boxes and class labels. This helps in understanding the model's performance visually.

## Results
The model's performance can be assessed through various metrics and visualizations. The results can be plotted using libraries like Matplotlib and Plotly.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

