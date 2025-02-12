import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.utils import load_img, img_to_array  # Updated import for load_img and img_to_array
from keras.models import load_model
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, precision_score

# Function to calculate PSNR
def calculate_psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:  # No noise in the image
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Load the low-quality image
input_image = cv2.imread(r'C:\Users\Lenovo\Downloads\pic10.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
gaussian_blur = cv2.GaussianBlur(input_image, (5, 5), 0)
enhanced_image_path = r'C:\Users\Lenovo\Downloads\enhanced_image_using_GussianBlur.jpg'
cv2.imwrite(enhanced_image_path, gaussian_blur)

# Display original and blurred images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(input_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Gaussian Blurred Image')
plt.imshow(gaussian_blur, cmap='gray')
plt.show()

# Load pre-trained ResNet50 model
resnet_model = ResNet50(weights='imagenet')
# Load and preprocess image  
img = load_img(enhanced_image_path, target_size=(224, 224)) 
x = img_to_array(img) 
x = np.expand_dims(x, axis=0) 
x = preprocess_input(x) 
# Predict objects 
resnet_predictions = resnet_model.predict(x) 
decoded_predictions = decode_predictions(resnet_predictions, top=3)[0] 
# Display predictions 
print("\nResNet50 Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i+1}: {label} ({score*100:.2f}%)")

# Load the YOLOv8 model
yolo_model = YOLO("yolov8n.pt")
# Predict objects
yolo_predictions = yolo_model.predict(enhanced_image_path)
print("\nYOLOv8 Predictions:")
for result in yolo_predictions:
    print(result.boxes)  # Print detection boxes
    result.show()  # Display the annotated image   
    result.save(filename=r'C:\Users\Lenovo\Downloads\object_PredictYOLO8.jpg')  # Save annotated image

# Calculate PSNR for ResNet50 and YOLO
psnr_resnet = calculate_psnr(input_image, gaussian_blur)
psnr_yolo = calculate_psnr(input_image, gaussian_blur)

# Simulate accuracy and precision scores for visualization (replace with actual values)
accuracy_resnet = 0.85  # Replace with actual accuracy
accuracy_yolo = 0.88  # Replace with actual accuracy
precision_resnet = 0.80  # Replace with actual precision
precision_yolo = 0.82  # Replace with actual precision

# Visualization of metrics
labels = ['ResNet50', 'YOLOv8']
accuracy_scores = [accuracy_resnet, accuracy_yolo]
psnr_scores = [psnr_resnet, psnr_yolo]
precision_scores = [precision_resnet, precision_yolo]

x = np.arange(len(labels))  # Label locations
width = 0.25  # Width of bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, accuracy_scores, width, label='Accuracy')
rects2 = ax.bar(x, psnr_scores, width, label='PSNR')
rects3 = ax.bar(x + width, precision_scores, width, label='Precision Rate')

# Add labels, title, and legend
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of ResNet50 and YOLOv8 on Accuracy, PSNR, and Precision Rate')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()