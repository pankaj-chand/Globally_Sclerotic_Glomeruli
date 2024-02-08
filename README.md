# Globally_Sclerotic_Glomeruli
Deep learning for binary classification of globally sclerotic glomeruli from 

## Overview
This project focuses on the binary classification of PAS stained microscopy images of Kidney Glomerulus using deep learning techniques. Glomerulus images are classified into two categories: Non-Globally Sclerotic (label 0) and Globally Sclerotic (label 1).

## Table of Contents
1. [Setting a side a Test set](#test-set)
2. [Inspection of the data](#inspection)
3. [Preprocessing of the data](#preprocessing)
4. [Padding](#white-padding)
5. [Approach](#approach)
6. [Implementation](#implementation)
7. [Instructions for Reproducibility](#instructions-for-reproducibility)
8. [Results](#results)

## Test Set

Usually, I would first split the data and set a test set aside until the models were trained.
However, since it was mentioned that a hold out set has already been created, I split the data and set the test set aside in my notebook code before starting to train the models.

## Inspection
Caveat: It is possible that multiple image files in the dataset come from a single patient. This was not mentioned in the instructions nor was it clear from inspecting the dataset. Hence, I treated each image file individually.

I found that the images in the dataset were already cropped as a rectangle to include only the glomerulus.

Images were of different sizes, and all were not square.

There was an imbalance of more than 1:4 with regard to the Positive:Negative class in the dataset as shown in the following figure.

![Imbalanced-Data](images/Imbalanced_Data.png)

## Preprocessing

(code is in 1_Preprocess_Data folder)

I did not intend to train any Fully convolutional Neural Networks, and other types of networks expect the input images to be of the same size.
 
Hence, I made the images square using: white-padding or zero-padding.

I resized the resulting images to common sizes such as: 128x128 and 224x224 (for pretrained models) and 512x512.

My code ensures that the glomerulus in the image is centered in both the Zero padding and White Padding variants.

## White-Padding

I needed to know whether to progress with the resized images containing white-padding or zero-padding.

Zero Padding:

Advantages:
Preserves the original intensity distribution of the image.
Can be beneficial when the background of the images is not uniform or does not have a consistent color.
Helps in avoiding the introduction of bias towards any specific intensity value in the image.

Disadvantages:
May introduce noise or irrelevant information when padding with zeros if the model is sensitive to the background information.
Could potentially increase computational cost during training and inference due to the larger input size.

White Padding:

Advantages:
Can help to maintain consistent background across images, which could be beneficial if the background carries irrelevant information for your task.
May lead to better generalization if the background is not informative for the classification task.

Disadvantages:
Risks introducing bias towards white color in the model's learning process, especially if the background of the original images is not white.
May cause loss of original intensity information, which could be important for certain tasks.
In the context of analyzing PAS stained images of kidney glomerulus, the choice between zero padding and white padding should be made based on empirical evaluation and domain knowledge. If the background information in the images is irrelevant or noisy, white padding might be preferred to provide a consistent background for the model to learn from. However, if maintaining the original intensity distribution is crucial for your task, zero padding could be a better choice.

Ultimately, it's recommended to experiment with both padding strategies and evaluate their performance on your specific deep learning task to determine which one works best for your dataset and model architecture.


## Approach

I made a baseline model using only one neuron with sigmoid activation function, and no hidden layers. This was basically Logistic Regression, and could only model linear relationships in the data. Yet, it got 90% accuracy on the test set.

Subsequently, I used a simplified version of the approach in the following publication.

"Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation," Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz-Melo, Jorge Samper-González, Alexandre Routier, Simona Bottani, Didier Dormont, Stanley Durrleman, Ninon Burgos, Olivier Colliot, Medical Image Analysis, Volume 63, 2020, 101694, ISSN 1361-8415.

1. Repeatedly add a Dense layer until the model overfits.

2. Replace the largest Dense layer with a CNN block

3. Go back to step 1 and repeat until there is no further improvement.


## Implementation
Provide details on how the code is organized and structured in the repository. Explain the purpose of each file or directory and how they contribute to the project.





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







