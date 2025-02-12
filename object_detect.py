# Importing the required libraries
import os
import random
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from ultralytics import YOLO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter
import seaborn as sns
import tqdm
import torch
import glob
import os
import json
from PIL import Image
import xml.etree.ElementTree as ET
import pprint
pp = pprint.PrettyPrinter(indent=4)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import torchvision.ops as ops
from torch.utils.data import distributed, RandomSampler, SequentialSampler
import random
import cv2
from collections import Counter
import kaggle
from ultralytics import YOLO

#  Authenticate with Kaggle
kaggle.api.authenticate()

dataset_path = 'C:/Users/Lenovo/underwater_dataset1'
# Check if the dataset is already downloaded
if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}, downloading...")
    # Download and unzip the dataset
    kaggle.api.dataset_download_files('slavkoprytula/aquarium-data-cots', path=dataset_path, unzip=True)
else:
    print(f"Dataset already exists at {dataset_path}. Skipping download.")

# Define the paths to the images and labels directories
train_images = dataset_path+'/aquarium_pretrain/train/images'
train_labels = dataset_path+'/aquarium_pretrain/train/labels'

val_images = dataset_path+'/aquarium_pretrain/valid/images'
val_labels = dataset_path+'/aquarium_pretrain/valid/labels'

test_images = dataset_path+'/aquarium_pretrain/test/images'
test_labels = dataset_path+'/aquarium_pretrain/test/labels'

# Define the labels
classes = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
Idx2Label = {idx: label for idx, label in enumerate(classes)}
Label2Index = {label: idx for idx, label in Idx2Label.items()}
print('Index to Label Mapping:', Idx2Label)
print('Label to Index Mapping:', Label2Index)


def visualize_image_with_annotation_bboxes(image_dir, label_dir):
    # Get list of all the image files in the directory
    image_files = sorted(os.listdir(image_dir))
    
    # Choose 6 random image files from the list
    sample_image_files = random.sample(image_files, 6)
    
    # Set up the plot with a single row
    fig, axs = plt.subplots(1, 6, figsize=(20, 5))  # Single row, increased figsize for larger images
    fig.patch.set_facecolor('black')  # Set the figure background to black
    
    # Loop over the random images and plot the bounding boxes
    for i, image_file in enumerate(sample_image_files):
        # Load the image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load the labels for this image
        label_path = os.path.join(label_dir, image_file[:-4] + '.txt')
        f = open(label_path, 'r')
        
        # Loop over the labels and plot the bounding boxes
        for label in f:
            class_id, x_center, y_center, width, height = map(float, label.split())
            h, w, _ = image.shape
            x_min = int((x_center - width/2) * w)
            y_min = int((y_center - height/2) * h)
            x_max = int((x_center + width/2) * w)
            y_max = int((y_center + height/2) * h)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, Idx2Label[int(class_id)], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    
        # Display the image in its subplot
        axs[i].imshow(image)
        axs[i].axis('off')  # Turn off axis for cleaner look

    # Add tight layout with minimal spacing
    plt.subplots_adjust(wspace=0.02, hspace=0)  # Minimal space between images
    plt.show()
   
# Visualize 6 sample images with bounding boxes
visualize_image_with_annotation_bboxes(train_images, train_labels)


# Read a image by path
image_path = os.path.join(train_images, os.listdir(train_images)[100])
image = cv2.imread(image_path)

# Get the size of the image
height, width, channels = image.shape
print('The image has dimensions {}x{} and {} channels'.format(height, width, channels))


# Load a pretrained nano model
model = YOLO('yolov8n.pt')

# free up GPU memory
torch.cuda.empty_cache()

# Define the path to the yaml data file
yaml_path = dataset_path+'/aquarium_pretrain/data.yaml'
# Training the model

def check_dataset_path(path):
    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist.")
    else:
        print(f"Path {path} exists and contains {len(os.listdir(path))} files.")

check_dataset_path(train_images)
check_dataset_path(val_images)
check_dataset_path(test_images)

model = YOLO("yolov8n.pt")

# results = model.train(
#     # data=r"C:\Users\Lenovo\AppData\Roaming\Ultralytics\underwater_dataset1\aquarium_pretrain\data.yaml",
#     data=r"C:\Users\Lenovo\underwater_dataset1\aquarium_pretrain\data.yaml",
#     epochs=1,
#     imgsz=(height, width, channels),
#     seed=42,
#     batch=16,
#     workers=4,
#     name='yolov8n_custom1'
# )

# # # Predict objects
predictions = model.predict(r"C:\Users\Lenovo\underwater_dataset1\aquarium_pretrain\test\images")

# %matplotlib inline
# read in the results.csv file as a pandas dataframe
df = pd.read_csv(r'C:\Users\Lenovo\OneDrive\Desktop\flask_website\runs\detect\yolov8n_custom5\results.csv')

df.columns = df.columns.str.strip()

# create subplots using seaborn
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

# plot the columns using seaborn
sns.lineplot(x='epoch', y='train/box_loss', data=df, ax=axs[0,0])
sns.lineplot(x='epoch', y='train/cls_loss', data=df, ax=axs[0,1])
sns.lineplot(x='epoch', y='train/dfl_loss', data=df, ax=axs[1,0])
sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=axs[1,1])
sns.lineplot(x='epoch', y='metrics/recall(B)', data=df, ax=axs[2,0])
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=axs[2,1])
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=df, ax=axs[3,0])
sns.lineplot(x='epoch', y='val/box_loss', data=df, ax=axs[3,1])
sns.lineplot(x='epoch', y='val/cls_loss', data=df, ax=axs[4,0])
sns.lineplot(x='epoch', y='val/dfl_loss', data=df, ax=axs[4,1])


# set titles and axis labels for each subplot
axs[0,0].set(title='Train Box Loss')
axs[0,1].set(title='Train Class Loss')
axs[1,0].set(title='Train DFL Loss')
axs[1,1].set(title='Metrics Precision (B)')
axs[2,0].set(title='Metrics Recall (B)')
axs[2,1].set(title='Metrics mAP50 (B)')
axs[3,0].set(title='Metrics mAP50-95 (B)')
axs[3,1].set(title='Validation Box Loss')
axs[4,0].set(title='Validation Class Loss')
axs[4,1].set(title='Validation DFL Loss')

# add suptitle and subheader
plt.suptitle('Training Metrics', fontsize=24)

# adjust top margin to make space for suptitle
plt.subplots_adjust(top=0.8)

# adjust spacing between subplots
plt.tight_layout()

plt.show()
# Adjust layout and display
# plt.tight_layout()
# plt.show()

# Loading the best performing model
model = YOLO(r"C:\Users\Lenovo\OneDrive\Desktop\flask_website\runs\detect\yolov8n_custom5\weights\best.pt")  # Load the best weights

# Evaluating the model on test dataset
metrics = model.val(conf=0.25, split='test')

print(f"Mean Average Precision @.5:.95 : {metrics.box.map}")    
print(f"Mean Average Precision @ .50   : {metrics.box.map50}") 
print(f"Mean Average Precision @ .70   : {metrics.box.map75}")

# Function to perform detections with trained model
def predict_detection(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Pass the image through the detection model and get the result
    detect_result = model(image)
    
    # Plot the detections
    detect_image = detect_result[0].plot()
    
    # Convert the image to RGB format
    detect_image = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)
    
    return detect_image

# Get list of all the image files in the test directory
image_files = sorted(os.listdir(test_images))
    
# Choose 12 random image files from the list
sample_image_files = random.sample(image_files, 12)
    
# Set up the plot
fig, axs = plt.subplots(4, 3, figsize=(15, 20))
    
# Loop over the random images and plot the detections of the trained model
for i, image_file in enumerate(sample_image_files):
    row = i // 3
    col = i % 3
        
    # Load the current image and run object detection
    image_path = os.path.join(test_images, image_file)
    detect_image = predict_detection(image_path)
    
    axs[row, col].imshow(detect_image)
    axs[row, col].axis('off')

plt.show()