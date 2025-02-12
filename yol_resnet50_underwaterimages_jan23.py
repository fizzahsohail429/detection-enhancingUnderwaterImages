import os
import random
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from ultralytics import YOLO
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import ToTensor
import matplotlib as pd

# Dataset paths
dataset_path = 'C:/Users/Lenovo/underwater_dataset1'
train_images = os.path.join(dataset_path, 'aquarium_pretrain/train/images')
train_labels = os.path.join(dataset_path, 'aquarium_pretrain/train/labels')
test_images = os.path.join(dataset_path, 'aquarium_pretrain/test/images')


def calculate_psnr_ssim(original_images, enhanced_images):
    psnr_values = []
    ssim_values = []

    for o, e in zip(original_images, enhanced_images):
        # Ensure both images are resized to the same dimensions (e.g., 256x256)
        o_resized = cv2.resize(o, (256, 256), interpolation=cv2.INTER_LINEAR)
        e_resized = cv2.resize(e, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Compute PSNR
        psnr_val = psnr(o_resized, e_resized, data_range=255)
        psnr_values.append(psnr_val)

        # Compute SSIM with an appropriate window size
        ssim_val = ssim(o_resized, e_resized, channel_axis=-1, win_size=7, data_range=255)
        ssim_values.append(ssim_val)

    return psnr_values, ssim_values

def plot_metric_graph(metric_values, metric_name):
    if not metric_values:  # Handle empty lists
        print(f"No data available for {metric_name}")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(metric_values, label=f'{metric_name} per Image')
    plt.title(f'{metric_name} for Enhanced Images')
    plt.xlabel('Image Index')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

# Utility functions
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# CNN-based enhancement (mock example)
def enhance_images_with_cnn(images):
    # Placeholder for actual CNN enhancement logic
    return [cv2.GaussianBlur(img, (5, 5), 0) for img in images]

# Load and enhance images
original_images = load_images_from_folder(train_images)
enhanced_images = enhance_images_with_cnn(original_images)

# Calculate and plot PSNR and SSIM
psnr_values, ssim_values = calculate_psnr_ssim(original_images, enhanced_images)
plot_metric_graph(psnr_values, "PSNR")
plot_metric_graph(ssim_values, "SSIM")

# YOLO-based object detection
model = YOLO('yolov8n.pt')

# Train YOLO model
results = model.train(
    data=os.path.join(dataset_path, r"C:\Users\Lenovo\underwater_dataset1\aquarium_pretrain\data.yaml"),
    epochs=1,
    imgsz=640,
    batch=16,
    workers=4,
    name='yolov8n_custom',
    save=True
    )

# Loading the best performing model
model = YOLO(r"C:\Users\Lenovo\OneDrive\Desktop\flask_website\runs\detect\yolov8n_custom5\weights\best.pt")  # Load the best weights

# Evaluate on test data
def evaluate_model_on_test(model, test_folder):
    precision_values, recall_values, f1_values = [], [], []
    for image_file in os.listdir(test_folder):
        image_path = os.path.join(test_folder, image_file)
        image = cv2.imread(image_path)
        result = model(image)

        # Check for detections
        detections = result[0].boxes.cpu().numpy()
        if len(detections) == 0:
            pred_labels = []  # Handle empty detections
        else:
            pred_labels = [int(box.cls) for box in result[0].boxes]

        # Placeholder for ground truth labels
        true_labels = []  # Replace with actual ground truth loading logic

        # Metrics calculation
        if true_labels:
            precision_values.append(precision_score(true_labels, pred_labels, average='weighted', zero_division=1))
            recall_values.append(recall_score(true_labels, pred_labels, average='weighted', zero_division=1))
            f1_values.append(f1_score(true_labels, pred_labels, average='weighted'))

    return precision_values, recall_values, f1_values


precision, recall, f1 = evaluate_model_on_test(model, test_images)
# plot_metric_graph([0.8, 0.85, 0.88], "Precision")
# plot_metric_graph([0.75, 0.8, 0.82], "Recall")
# plot_metric_graph([0.78, 0.83, 0.85], "F1-Score")

# Plot precision, recall, and F1 score
plot_metric_graph(precision, "Precision")
plot_metric_graph(recall, "Recall")
plot_metric_graph(f1, "F1 Score")

# mAP plotting
import pandas 
results_df = pandas.read_csv(os.path.join('C:/Users/Lenovo/OneDrive/Desktop/flask_website/runs/detect/yolov8n_custom5', 'results.csv'))
plt.figure(figsize=(10, 5))
plt.plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label="mAP@50")
plt.title("mAP@50 Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("mAP@50")
plt.legend()
plt.show()
