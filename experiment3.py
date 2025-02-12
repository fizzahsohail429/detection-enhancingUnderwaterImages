# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
# from tensorflow.keras.utils import load_img, img_to_array  # Updated import for load_img and img_to_array
# from keras.models import load_model
# from ultralytics import YOLO
# from sklearn.metrics import classification_report, confusion_matrix
# # Load the low-quality image
# input_image = cv2.imread(r'C:\Users\Lenovo\Downloads\pic12.jpg', cv2.IMREAD_GRAYSCALE)


# # Apply Gaussian Blur
# gaussian_blur = cv2.GaussianBlur(input_image, (5, 5), 0)
# enhanced_image_path = r'C:\Users\Lenovo\Downloads\enhanced_image_using_GussianBlur.jpg'
# cv2.imwrite(enhanced_image_path, gaussian_blur)

# # Display original and blurred images
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(input_image, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.title('Gaussian Blurred Image')
# plt.imshow(gaussian_blur, cmap='gray')
# plt.show()

# # Load pre-trained ResNet50 model
# model = ResNet50(weights='imagenet')
# # Load and preprocess image  
# img = load_img(enhanced_image_path, target_size=(224, 224)) 
# x = img_to_array(img) 
# x = np.expand_dims(x, axis=0) 
# x = preprocess_input(x) 
# # Predict objects 
# predictions = model.predict(x) 
# decoded_predictions = decode_predictions(predictions, top=3)[0] 
# # Display predictions 
# for i, (imagenet_id, label, score) in enumerate(decoded_predictions): print(f"{i+1}: {label} ({score*100:.2f}%)")

# # Load the YOLOv8 model
# model = YOLO("yolov8n.pt")
# # Predict objects
# predictions = model.predict(enhanced_image_path)
# for result in predictions:
#      print(result.boxes)  # Print detection boxes
#      result.show()  # Display the annotated image   
#      result.save(filename=r'C:\Users\Lenovo\Downloads\object_PredictYOLO8.jpg')  # Save annotated image

#################################################################
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
# from tensorflow.keras.utils import load_img, img_to_array  # Updated import for load_img and img_to_array
# from keras.models import load_model
# from ultralytics import YOLO
# from sklearn.metrics import classification_report, confusion_matrix, precision_score

# # Function to calculate PSNR
# def calculate_psnr(original, enhanced):
#     mse = np.mean((original - enhanced) ** 2)
#     if mse == 0:  # No noise in the image
#         return float('inf')
#     max_pixel = 255.0
#     psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
#     return psnr

# # Load the low-quality image
# input_image = cv2.imread(r'C:\Users\Lenovo\Downloads\pic12.jpg', cv2.IMREAD_GRAYSCALE)

# # Apply Gaussian Blur
# gaussian_blur = cv2.GaussianBlur(input_image, (5, 5), 0)
# enhanced_image_path = r'C:\Users\Lenovo\Downloads\enhanced_image_using_GussianBlur.jpg'
# cv2.imwrite(enhanced_image_path, gaussian_blur)

# # Display original and blurred images
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(input_image, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.title('Gaussian Blurred Image')
# plt.imshow(gaussian_blur, cmap='gray')
# plt.show()

# # Load pre-trained ResNet50 model
# resnet_model = ResNet50(weights='imagenet')
# # Load and preprocess image  
# img = load_img(enhanced_image_path, target_size=(224, 224)) 
# x = img_to_array(img) 
# x = np.expand_dims(x, axis=0) 
# x = preprocess_input(x) 
# # Predict objects 
# resnet_predictions = resnet_model.predict(x) 
# decoded_predictions = decode_predictions(resnet_predictions, top=3)[0] 
# # Display predictions 
# print("\nResNet50 Predictions:")
# for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
#     print(f"{i+1}: {label} ({score*100:.2f}%)")

# # Load the YOLOv8 model
# yolo_model = YOLO("yolov8n.pt")
# # Predict objects
# yolo_predictions = yolo_model.predict(enhanced_image_path)
# print("\nYOLOv8 Predictions:")
# for result in yolo_predictions:
#     print(result.boxes)  # Print detection boxes
#     result.show()  # Display the annotated image   
#     result.save(filename=r'C:\Users\Lenovo\Downloads\object_PredictYOLO8.jpg')  # Save annotated image

# # Calculate PSNR for ResNet50 and YOLO
# psnr_resnet = calculate_psnr(input_image, gaussian_blur)
# psnr_yolo = calculate_psnr(input_image, gaussian_blur)

# # Simulate accuracy and precision scores for visualization (replace with actual values)
# accuracy_resnet = 0.85  # Replace with actual accuracy
# accuracy_yolo = 0.88  # Replace with actual accuracy
# precision_resnet = 0.80  # Replace with actual precision
# precision_yolo = 0.82  # Replace with actual precision

# # Visualization of metrics
# labels = ['ResNet50', 'YOLOv8']
# accuracy_scores = [accuracy_resnet, accuracy_yolo]
# psnr_scores = [psnr_resnet, psnr_yolo]
# precision_scores = [precision_resnet, precision_yolo]

# x = np.arange(len(labels))  # Label locations
# width = 0.25  # Width of bars

# fig, ax = plt.subplots(figsize=(10, 6))
# rects1 = ax.bar(x - width, accuracy_scores, width, label='Accuracy')
# rects2 = ax.bar(x, psnr_scores, width, label='PSNR')
# rects3 = ax.bar(x + width, precision_scores, width, label='Precision Rate')

# # Add labels, title, and legend
# ax.set_xlabel('Models')
# ax.set_ylabel('Scores')
# ax.set_title('Comparison of ResNet50 and YOLOv8 on Accuracy, PSNR, and Precision Rate')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# # Display the plot
# plt.tight_layout()
# plt.show()
##############################################

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.applications.resnet import ResNet50, preprocess_input
# from tensorflow.keras.utils import load_img, img_to_array  # Updated import for load_img and img_to_array
# from ultralytics import YOLO

# # Load the low-quality image
# input_image = cv2.imread(r'C:\Users\Lenovo\Downloads\pic12.jpg', cv2.IMREAD_GRAYSCALE)

# # Apply Gaussian Blur (Optional preprocessing step)
# gaussian_blur = cv2.GaussianBlur(input_image, (5, 5), 0)
# blurred_image_path = r'C:\Users\Lenovo\Downloads\blurred_image.jpg'
# cv2.imwrite(blurred_image_path, gaussian_blur)

# # Display original and blurred images
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(input_image, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.title('Gaussian Blurred Image')
# plt.imshow(gaussian_blur, cmap='gray')
# plt.show()

# # Load pre-trained ResNet50 for feature extraction
# resnet_model = ResNet50(weights='imagenet', include_top=False)  # Exclude the fully connected layer
# # Load and preprocess image
# img = load_img(blurred_image_path, target_size=(224, 224))  # Resize to ResNet50 input size
# x = img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# # Extract features using ResNet50
# features = resnet_model.predict(x)
# print(f"Extracted Features Shape: {features.shape}")

# # Convert features to a format YOLO can process (resize and normalize)
# # Select the first feature map and resize it to an image-like format
# processed_image = features[0].mean(axis=-1)  # Average across channels to get a 2D image
# processed_image = cv2.resize(processed_image, (224, 224))  # Resize to match YOLO's input
# processed_image = np.uint8((processed_image - processed_image.min()) / 
#                            (processed_image.max() - processed_image.min()) * 255)  # Normalize to 0-255
# processed_image_path = r'C:\Users\Lenovo\Downloads\processed_image.jpg'
# cv2.imwrite(processed_image_path, processed_image)


# # Load the YOLOv8 model
# yolo_model = YOLO("yolov8n.pt")

# # Predict objects using YOLOv8 on the processed image
# yolo_predictions = yolo_model.predict(processed_image_path)
# print("\nYOLOv8 Predictions:")
# for result in yolo_predictions:
#     print(result.boxes)  # Print detection boxes
#     result.show()  # Display the annotated image
#     result.save(filename=r'C:\Users\Lenovo\Downloads\object_predictions.jpg')  # Save annotated image



#########################

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.applications.resnet import ResNet50, preprocess_input
# from tensorflow.keras.utils import load_img, img_to_array
# from ultralytics import YOLO

# # Load the low-quality image
# input_image = cv2.imread(r'C:\Users\Lenovo\Downloads\pic10.jpg', cv2.IMREAD_GRAYSCALE)

# # Apply Gaussian Blur (Optional preprocessing step)
# gaussian_blur = cv2.GaussianBlur(input_image, (5, 5), 0)
# blurred_image_path = r'C:\Users\Lenovo\Downloads\blurred_image.jpg'
# cv2.imwrite(blurred_image_path, gaussian_blur)

# # Display original and blurred images
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(input_image, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.title('Gaussian Blurred Image')
# plt.imshow(gaussian_blur, cmap='gray')
# plt.show()

# # Load pre-trained ResNet50 for feature extraction
# resnet_model = ResNet50(weights='imagenet', include_top=False)  # Exclude the fully connected layer
# # Load and preprocess image
# img = load_img(blurred_image_path, target_size=(224, 224))  # Resize to ResNet50 input size
# x = img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# # Extract features using ResNet50
# features = resnet_model.predict(x)
# print(f"Extracted Features Shape: {features.shape}")

# # Enhance feature map for YOLO
# # Select the first feature map layer
# enhanced_features = features[0, :, :, 0]  # Use the first channel for simplicity
# enhanced_features = cv2.resize(enhanced_features, (224, 224))  # Resize to match YOLO's input
# enhanced_features = np.uint8((enhanced_features - enhanced_features.min()) / 
#                              (enhanced_features.max() - enhanced_features.min()) * 255)  # Normalize to 0-255

# # Blend enhanced features with the original image
# blended_image = cv2.addWeighted(cv2.resize(input_image, (224, 224)), 0.5, enhanced_features, 0.5, 0)
# blended_image_path = r'C:\Users\Lenovo\Downloads\blended_image.jpg'
# cv2.imwrite(blended_image_path, blended_image)

# # Display blended image
# plt.title("Blended Image")
# plt.imshow(blended_image, cmap='gray')
# plt.show()

# # Load the YOLOv8 model
# yolo_model = YOLO("yolov8n.pt")

# # Predict objects using YOLOv8 on the blended image
# yolo_predictions = yolo_model.predict(blended_image_path)
# print("\nYOLOv8 Predictions:")
# for result in yolo_predictions:
#     print(result.boxes)  # Print detection boxes
#     result.show()  # Display the annotated image
#     result.save(filename=r'C:\Users\Lenovo\Downloads\object_predictions.jpg')  # Save annotated image



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


# Imports and paths remain the same as your original code

def calculate_psnr_ssim(original_images, enhanced_images):
    psnr_values = []
    ssim_values = []

    for o, e in zip(original_images, enhanced_images):
        o_resized = cv2.resize(o, (256, 256), interpolation=cv2.INTER_LINEAR)
        e_resized = cv2.resize(e, (256, 256), interpolation=cv2.INTER_LINEAR)

        psnr_val = psnr(o_resized, e_resized, data_range=255)
        psnr_values.append(psnr_val)

        ssim_val = ssim(o_resized, e_resized, channel_axis=-1, win_size=7, data_range=255)
        ssim_values.append(ssim_val)

    return psnr_values, ssim_values

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def enhance_images_with_cnn(images):
    return [cv2.GaussianBlur(img, (5, 5), 0) for img in images]

original_images = load_images_from_folder(train_images)
enhanced_images = enhance_images_with_cnn(original_images)

# PSNR and SSIM calculation
psnr_values, ssim_values = calculate_psnr_ssim(original_images, enhanced_images)
print(f"Average PSNR: {np.mean(psnr_values):.2f}")
print(f"Average SSIM: {np.mean(ssim_values):.4f}")

# YOLO-based object detection
model = YOLO('yolov8n.pt')

results = model.train(
    data=os.path.join(dataset_path, r"C:\Users\Lenovo\underwater_dataset1\aquarium_pretrain\data.yaml"),
    epochs=1,
    imgsz=640,
    batch=16,
    workers=4,
    name='yolov8n_custom',
    save=True
)

def evaluate_model_on_test(model, test_folder):
    precision_values, recall_values, f1_values = [], [], []
    for image_file in os.listdir(test_folder):
        image_path = os.path.join(test_folder, image_file)
        image = cv2.imread(image_path)
        result = model(image)

        detections = result[0].boxes.cpu().numpy()
        pred_labels = [int(box.cls) for box in result[0].boxes] if len(detections) > 0 else []

        true_labels = []  # Replace with actual ground truth loading logic

        if true_labels:
            precision_values.append(precision_score(true_labels, pred_labels, average='weighted', zero_division=1))
            recall_values.append(recall_score(true_labels, pred_labels, average='weighted', zero_division=1))
            f1_values.append(f1_score(true_labels, pred_labels, average='weighted'))

    return precision_values, recall_values, f1_values

precision, recall, f1 = evaluate_model_on_test(model, test_images)

# Print metrics
if precision:
    print(f"Average Precision: {np.mean(precision):.4f}")
if recall:
    print(f"Average Recall: {np.mean(recall):.4f}")
if f1:
    print(f"Average F1-Score: {np.mean(f1):.4f}")
