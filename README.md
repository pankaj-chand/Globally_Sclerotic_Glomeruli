# Globally_Sclerotic_Glomeruli
Deep learning for binary classification of globally sclerotic glomeruli from 

## Overview
This project focuses on the binary classification of PAS stained microscopy images of Kidney Glomerulus using deep learning techniques. Glomerulus images are classified into two categories: Non-Globally Sclerotic (label 0) and Globally Sclerotic (label 1).

## Table of Contents
1. [Setting aside a Test set](#test-set)
2. [Inspection of the data](#inspection)
3. [Preprocessing of the data](#preprocessing)
4. [Padding](#type-of-padding)
5. [Approach and baseline models](#approach)
6. [Implementation](#implementation)
7. [Homemade CNN models 97+% accuracy](#homemade-models)
8. [Large Pretrained Models 99+% accuracy](#large-pretrained-models)
9. [Instructions for Reproducibility](#instructions-for-reproducibility)
10. [Results](#results)

IMPORTANT: Only students who have a faculty sponsor can access HiperGator premium resources. Students who are onlhy taking a course that uses HiperGator would have access to limited HiperGator resources. Since I do not have a facuty sponsor nor are any of my current courses using HiperGator, I do not have access to HiperGator. So, I have used only Google Collab Pro. 

## Test Set

Usually, I would first split the data and set a test set aside until the models were trained.
However, since it was mentioned that a hold out set has already been created by Dr. Paul and Dr. Naglah, so I split the data and set my own test set aside in my notebook code after inspecting the data, but before starting to train the models. The data was split pseudo-randomly using five different seeds to repeat the experiments five times and mimic Scikit-Learn's 5-Fold Cross Validation. However, only one of the seeds is given in the code.

## Inspection

Caveat: It is possible that multiple image files in the dataset come from a single whole slide image or from a single patient or subject. This was not mentioned in the instructions nor was it clear from inspecting the dataset. Hence, I treated each image file individually. However, this most probably led to data leakage. If you can let me know how exactly multiple images can be identified to belong to the same subject, then I would need to split the dataset on the subject-level and retrain the models.

### Dataset:

a. I found that the images in the dataset were already cropped as a rectangle to include only the glomerulus, i.e, the region of interest.

b. Images were of different sizes, and all were not square.

c. There were 5758 images in the entire dataset.

d. There was an imbalance of more than 1:4 with regard to the Positive:Negative class in the dataset as shown in the following figure. I would address the imbalance only if required after seeing the performance of the trained models.

![Imbalanced-Data](images/Imbalanced_Data.png)

## Preprocessing

### Code is in 1_Preprocess_Data folder

I did not intend to train Fully convolutional Neural Networks, and other types of networks expect the input images to be of the same size.
 
Hence, I made the images square using: white-padding or zero-padding.

I resized the resulting images to common sizes such as: 128x128 and 224x224 (for pretrained models) and 512x512.

My code ensures that the glomerulus in the image is centered in both the Zero padding and White Padding variants.

## Type of Padding

I needed to know whether to progress with the resized images containing white-padding or zero-padding.

### Zero Padding:

Advantages:
Preserves the original intensity distribution of the image.
Can be beneficial when the background of the images is not uniform or does not have a consistent color.
Helps in avoiding the introduction of bias towards any specific intensity value in the image.

Disadvantages:
May introduce noise or irrelevant information when padding with zeros if the model is sensitive to the background information.
Could potentially increase computational cost during training and inference due to the larger input size.

### White Padding:

Advantages:
Can help to maintain consistent background across images, which could be beneficial if the background carries irrelevant information for your task.
May lead to better generalization if the background is not informative for the classification task.

Disadvantages:
Risks introducing bias towards white color in the model's learning process, especially if the background of the original images is not white.
May cause loss of original intensity information, which could be important for certain tasks.
In the context of analyzing PAS stained images of kidney glomerulus, the choice between zero padding and white padding should be made based on empirical evaluation and domain knowledge. If the background information in the images is irrelevant or noisy, white padding might be preferred to provide a consistent background for the model to learn from. However, if maintaining the original intensity distribution is crucial for your task, zero padding could be a better choice.

Ultimately, it's recommended to experiment with both padding strategies and evaluate their performance on your specific deep learning task to determine which one works best for your dataset and model architecture.

### Images from the folder 2_WhitePad_OR_ZeroPad

Since I had already built a PyTorch model with the glomerulus images that Sam had kindly given me two weeks back, I trained that same model for 20 epochs each on both the zero-padded and white-padded images, and at dimensions of 128x128 pixels and 224x224 pixels. I have not provided the code of that basic preliminary model, but I may be willing to provide it in a sequel. The following figures show the preliminary results.

#### Zero padding 128x128 pixels
![zero-padding_128x128_20 epochs](2_WhitePad_OR_ZeroPad/ZeroPad_128_20_graphs.png)
![zero-padding_128x128_20 epochs_cm](2_WhitePad_OR_ZeroPad/ZeroPad_128_20_confusion_matrix.png)

#### White padding 128x128 pixels
![white-padding_128x128_20 epochs](2_WhitePad_OR_ZeroPad/WhitePad_128_20_graphs.png)
![white-padding_128x128_20 epochs_cm](2_WhitePad_OR_ZeroPad/WhitePad_128_20_confusion_matrix.png)

#### Zero padding 224x224 pixels
![zero-padding_224x224_20 epochs](2_WhitePad_OR_ZeroPad/ZeroPad_224_20_graphs.png)
![zero-padding_224x224_20 epochs_cm](2_WhitePad_OR_ZeroPad/ZeroPad_224_20_confusion_matrix.png)

#### White padding 224x224 pixels
![white-padding_224x224_20 epochs](2_WhitePad_OR_ZeroPad/WhitePad_224_20_graphs.png)
![white-padding_224x224_20 epochs_cm](2_WhitePad_OR_ZeroPad/WhitePad_224_20_confusion_matrix.png)

Conclusion: Except for the sudden dip in the middle of the last validation curve, I found that the white padding provided more stable training and validation curves. Hence, I decided to continue using only the White-Padded version of pre-processed images.

Since I am using Google Collab Pro, not HiperGator, usable RAM is around 20 GB, and so I decided not to use the 512x512 resized images due to low RAM.

The preliminary results show that 128x128 sized images were providing better results than the 224x224 images. Since I was already getting 97% accuracy with a simple model, I thought there would be no point in using large architectures or pretrained models that prefer 224x224 sized images. However, later I decided to use 224x224 pixel images with the large architectures.

## Approach

### The experiments, code and model summaries for this section is in the notebook in folder 3_Baseline_Models

I made a baseline model using only one neuron with sigmoid activation function, and no hidden layers. This was basically Logistic Regression, and could only model linear relationships in the data. Yet, it got 90% accuracy on the test set.

![image](https://github.com/pankaj-chand/Globally_Sclerotic_Glomeruli/assets/49002748/e6e4a4f2-494c-48de-8d4c-f4a2132bc4ed)
![model1](images/baseline_models/Model1_NoHiddenLayers.png)
#### No Hidden Layers (49152 = 128x128x3) Test result: 89.601 loss: 0.244

Subsequently, I used a simplified version of the approach in the following publication.

"Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation," Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz-Melo, Jorge Samper-González, Alexandre Routier, Simona Bottani, Didier Dormont, Stanley Durrleman, Ninon Burgos, Olivier Colliot, Medical Image Analysis, Volume 63, 2020, 101694, ISSN 1361-8415.

Step 1. Repeatedly add a Dense layer until the model overfits.

Step 2. Replace the largest Dense layer with a Convolutional block

Step 3. Go back to step 1 and repeat until there is no further improvement.

Step 4. Add another Convolutional block to make the model learn  more useful features.

Step 5. Go back to step 1 and repeat until there is no further improvement.

![image](https://github.com/pankaj-chand/Globally_Sclerotic_Glomeruli/assets/49002748/43e71896-77a6-4cb4-b71c-482d840827f4)
![model2](images/baseline_models/Model2.png)
#### One Large Dense Hidden Layer (49152 = 128x128x3) Test result: 92.721 loss: 0.224

![image](https://github.com/pankaj-chand/Globally_Sclerotic_Glomeruli/assets/49002748/9c69368f-8217-4331-889a-a92b9e955a30)
![model3](images/baseline_models/Model3.png)
#### One Small Dense Hidden Layer (49152 = 128x128x3) Test result: 91.854 loss: 0.213

![image](https://github.com/pankaj-chand/Globally_Sclerotic_Glomeruli/assets/49002748/f0343e3f-a801-43cd-92a1-977c444da521)
![model4](images/baseline_models/Model4.png)
#### Two Large Dense Hidden Layers (49152 = 128x128x3) Test result: 92.721 loss: 0.166

Since the architecture was not able to learn enough, I added a CNN layer.

![image](https://github.com/pankaj-chand/Globally_Sclerotic_Glomeruli/assets/49002748/405b1e74-6c7f-44e7-b39d-580afdbdf3d4)
![model5](images/baseline_models/Model5.png)
#### One CNN Layer and Two Large Dense Hidden Layers (49152 = 128x128x3) Test result: 96.534 loss: 0.102

The model is learning more. So, I added another Dense layer to make it overfit.

![image](https://github.com/pankaj-chand/Globally_Sclerotic_Glomeruli/assets/49002748/3d7d22be-a45f-4c5a-aec1-3b4467626118)
![model6](images/baseline_models/Model6.png)
#### One CNN Layer and Three Dense Hidden Layers (49152 = 128x128x3) Test result: 96.534 loss: 0.107

I added another Convolutional layer to make the model learn more features.

![image](https://github.com/pankaj-chand/Globally_Sclerotic_Glomeruli/assets/49002748/333e018e-5068-43ea-bbbf-3a9a4da4cfb2)
![model7](images/baseline_models/Model7.png)
#### Two CNN Layers (layer 2 has double the filters) and Three Dense Hidden Layers (49152 = 128x128x3) Test result: 96.187 loss: 0.101

![image](https://github.com/pankaj-chand/Globally_Sclerotic_Glomeruli/assets/49002748/333e018e-5068-43ea-bbbf-3a9a4da4cfb2)
![model8](images/baseline_models/Model8.png)
#### To smoothen the accuracy curves, I added Batch Normalization. Test result: 95.494 loss: 0.115
#### The curves show that Batch Normalization helps the model to learn more on the training data . So, the model is finally overfitting.


![image](https://github.com/pankaj-chand/Globally_Sclerotic_Glomeruli/assets/49002748/333e018e-5068-43ea-bbbf-3a9a4da4cfb2)
![model9](images/baseline_models/Model9.png)
#### I added MaxPooling to see if pooling would help. The validation accuracy curves show that MaxPooling was not useful. Test result: 95.147 loss: 0.200

![image](https://github.com/pankaj-chand/Globally_Sclerotic_Glomeruli/assets/49002748/333e018e-5068-43ea-bbbf-3a9a4da4cfb2)
![model10](images/baseline_models/Model10.png)
#### Two CNN Layers (layer 2 has double the filters) and Three Dense Hidden Layers (49152 = 128x128x3) Test result: 94.627 loss: 0.207


## Implementation

After December 2023, I do NOT have access to HiperGator.
So, I am using Google Collab Pro which is limited to 20 GB  of usable RAM.
Hence, I have not used SVM or PCA on the entire image dataset because these techniques usually require more than 20  GB of RAM for the given dataset.
I have limited the size of the images to 224x224 pixels

### Data Augmentation
In the baseline and homemade models, I am using a slight amount of data augmentation using the ImageDataGenerator from the Keras framework to make the model robust and generalizable. The data augmentation techniques are basically:

a. Horizontal Flip
b. Rotation upto 15 degrees
c. Width shift upto 10%
d. Height Shift upto 10%

### TensorFlow
The baseline and homemade models are coded in TensorFlow Keras because the code is more readable and makes it easier to explain my approach. They use CNN and Dense layers.

### PyTorch
The large and pretrained models are coded in PyTorch because PyTorch framework usually gives better performance with lower training times.I have tried VGG16, VGG19 and ResNet18, with all layers pretrained on ImageNet and frozen, and all layers pretrained on ImageNet and retrained on the dataset.


## Homemade Models


## Large Pretrained Models


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


## My models

1. Best (accuracy) model (VGG19) link on Google Drive: https://drive.google.com/file/d/1cxtIapDT08ral7OVUep0DbkCO-Rcb_5n/view?usp=sharing

2. Fastest training model (RESNET18) link on Google Drive: https://drive.google.com/file/d/1JDnPUlrsVBvTrylPDCF0VlzThSUgmD73/view?usp=sharing

3. Medium accuracy and medium training time model (VGG16) link on Google Drive: https://drive.google.com/file/d/17wppOyLeqQOJFBpiTQShrbPvuHtHDjYu/view?usp=sharing

## Example Image
![Glomerulus Image](images/glomerulus_example.png)







