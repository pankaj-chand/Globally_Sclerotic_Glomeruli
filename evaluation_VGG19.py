import os
import sys
import csv
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from glob import glob


def white_pad_to_square(image_path):
    # Open the image
    image = Image.open(image_path)

    # Find the larger dimension
    max_dim = max(image.size)

    # Create a new square image with white background
    new_image = Image.new("RGB", (max_dim, max_dim), "white")

    # Paste the original image onto the new white image
    offset = ((max_dim - image.width) // 2, (max_dim - image.height) // 2)
    new_image.paste(image, offset)

    return new_image

def resize_image(image, target_size):
    # Resize the image to the target size
    resized_image = image.resize((target_size, target_size), Image.ANTIALIAS)

    return resized_image

TARGET_SIZE = 224  # Change this to your desired size


# Define the CNNModel class according to your model architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)

        # Replace output layer according to our problem
        in_feats = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(in_feats, 2)

    def forward(self, x):
        x = self.vgg19(x)
        return x

# Function to load the model
def load_model(model_path, device):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


# Function to process an image
def process_image(image_path):
    # Define image transformations
    preprocess = transforms.Compose([
                      transforms.Resize((224, 224)),
                      transforms.ToTensor()
    ])
    # Open and preprocess the image
    image = Image.open(image_path)
    image = preprocess(image)
    return image

# Function to perform evaluation on images in a folder
def evaluate_folder(folder_path, model, device):
    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Create a CSV file for storing evaluation results
    with open('evaluation.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['name', 'ground truth'])  # Header row
        # Iterate over images and perform inference
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = process_image(image_path)
            image = image.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image)
            # Get the predicted class
            prediction = torch.argmax(output, dim=1).item()
            # Write the result to the CSV file
            csv_writer.writerow([image_file, prediction])

if __name__ == "__main__":
    # Set device (cuda if available, else cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model_path = '/content/vgg19_all_layers_9_epochs_ver2.pth'
    model = load_model(model_path, device)

    # Check if a folder path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python evaluation.py /path/to/images_folder")
        sys.exit(1)

    # Get the folder path from the command-line argument
    folder_path = sys.argv[1]

    print(f'Folder Path is {folder_path}')


    # Create a folder for storing preprocessed images if it doesn't exist
    preprocessed_folder = "Preprocessed_Images"
    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)

    # Preprocess images
    # Process each image in the input folder
    for filename in os.listdir(folder_path):

      input_path = os.path.join(folder_path, filename)

      output_filename = filename

      output_path = os.path.join(preprocessed_folder, output_filename)

      # White pad the image to make it square
      padded_image = white_pad_to_square(input_path)

      # Resize the image to specific square dimensions
      resized_image = resize_image(padded_image, TARGET_SIZE)

      # Save the resized image
      resized_image.save(output_path)


    # Perform evaluation and generate CSV
    evaluate_folder(preprocessed_folder, model, device)

    print("Evaluation completed. Evaluation results saved in evaluation.csv.")
