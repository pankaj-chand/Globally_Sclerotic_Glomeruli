# Globally_Sclerotic_Glomeruli
Deep learning for binary classification of globally sclerotic glomeruli from 

## Overview
This project focuses on the binary classification of PAS stained microscopy images of Kidney Glomerulus using deep learning techniques. Glomerulus images are classified into two categories: Non-Globally Sclerotic (label 0) and Globally Sclerotic (label 1).

## Table of Contents
1. [Exploration](#exploration)
2. [Approach](#approach)
3. [Implementation](#implementation)
4. [Instructions for Reproducibility](#instructions-for-reproducibility)
5. [Results](#results)

## Exploration
Provide an overview of the data exploration phase. This may include insights gained from exploring the dataset, data preprocessing steps, and visualizations.

## Approach
Describe the approach taken to build and train the deep learning model. This should include details such as the choice of model architecture, loss function, optimization algorithm, and any other relevant hyperparameters.

## Implementation
Provide details on how the code is organized and structured in the repository. Explain the purpose of each file or directory and how they contribute to the project.

|-- data/
| |-- train/
| |-- test/
|-- models/
| |-- model.py
|-- notebooks/
| |-- data_exploration.ipynb
| |-- model_training.ipynb
|-- README.md
|-- requirements.txt
|-- train.py
|-- evaluate.py



## Instructions for Reproducibility
Include step-by-step instructions for reproducing the results of the project. This should cover environment setup, data preprocessing, model training, and evaluation.

1. **Clone the repository:**

git clone https://github.com/yourusername/your-repository.git
cd your-repository


2. **Install dependencies:**

pip install -r requirements.txt


3. **Prepare the data:**

Download the dataset and place it in the 'data/' directory
Preprocess the data if necessary


4. **Train the model:**

python train.py --data_path data/train --epochs 50 --batch_size 32


5. **Evaluate the model:**

python evaluate.py --data_path data/test --model_path models/model.pth


## Results
Summarize the results of the classification task. Include metrics such as accuracy, precision, recall, and F1-score, as well as any qualitative observations.

## Example Image
![Glomerulus Image](images/glomerulus_example.png)







