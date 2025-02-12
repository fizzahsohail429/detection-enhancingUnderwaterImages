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

# Path where the dataset will be downloaded and extracted
# dataset_path = './underwater_dataset'
# kaggle.api.dataset_download_files('slavkoprytula/aquarium-data-cots', path=dataset_path, unzip=True)
dataset_path = 'C:/Users/Lenovo/underwater_dataset1'

# Dataset folder paths
TRAIN_PATH = os.path.join(dataset_path, 'aquarium_pretrain', 'train')
VAL_PATH = os.path.join(dataset_path, 'aquarium_pretrain', 'valid')
TEST_PATH = os.path.join(dataset_path, 'aquarium_pretrain', 'test')

TRAIN_LABELS = os.path.join(TRAIN_PATH, 'labels')
VAL_LABELS = os.path.join(VAL_PATH, 'labels')
TEST_LABELS = os.path.join(TEST_PATH, 'labels')

TRAIN_IMAGES = os.path.join(TRAIN_PATH, 'images')
VAL_IMAGES = os.path.join(VAL_PATH, 'images')
TEST_IMAGES = os.path.join(TEST_PATH, 'images')

# Verify dataset structure
def describe_dataset(image_path, label_path):
    try:
        images = os.listdir(image_path)
        labels = os.listdir(label_path)
        print(f"Number of images: {len(images)}")
        print(f"Number of labels: {len(labels)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")

describe_dataset(TRAIN_IMAGES, TRAIN_LABELS)

def class_distribution(label_path):
    label_counts = Counter()
    for label_file in os.listdir(label_path):
        with open(os.path.join(label_path, label_file)) as f:
            for line in f.readlines():
                class_id = int(line.split()[0])  # Assuming YOLO format
                label_counts[class_id] += 1
    return label_counts

train_distribution = class_distribution(TRAIN_LABELS)
print("Class Distribution:", train_distribution)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


import albumentations as A
from albumentations.pytorch import ToTensorV2

augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.CLAHE(p=0.2),
    A.Resize(640, 640),
    ToTensorV2()
])

class UnderwaterDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, augmentation=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt'))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels (YOLO format: class_id x_center y_center width height)
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.split())
                boxes.append([class_id, x, y, w, h])

        if self.augmentation:
            augmented = self.augmentation(image=image, bboxes=boxes)
            image, boxes = augmented['image'], augmented['bboxes']

        if self.transform:
            image = self.transform(image)

        return image, boxes
    


# model = YOLO("yolov5s.pt") 
model = YOLO('runs/detect/yolov8n_custom5/weights/best.pt')  # Load the best weights
# model.train(data=r'c:\Users\Lenovo\underwater_dataset\aquarium_pretrain\data.yaml', epochs=50, imgsz=640)  # Train on your dataset
results = model.predict(source=TEST_IMAGES, save=True)  # Run inference
print (results,'!!!!!!!!!!!!!!!!!!!!')

