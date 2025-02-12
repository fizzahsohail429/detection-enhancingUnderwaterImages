import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r'C:\Users\Lenovo\Downloads\blurred_images.jpg')

# Define a sharpening kernel
sharpening_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

# Apply the sharpening filter
sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

# Display the original and sharpened images
# cv2.imshow('Original Image', image)
# cv2.imshow('Sharpened Image', sharpened_image)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Sharpened Image")
plt.imshow(sharpened_image)
plt.axis('off')

plt.show()
# # Wait for a key press and close the image windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # # # #######################################################################3
# # # # import cv2
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt

# # # # # Load the image
# # # # image = cv2.imread(r'C:\Users\Lenovo\Downloads\blurred_images.jpg', cv2.IMREAD_COLOR)

# # # # def add_salt_and_pepper_noise(img, salt_prob, pepper_prob):
# # # #     # Ensure the image is in color
# # # #     if len(img.shape) == 2:  # Grayscale image
# # # #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
# # # #     noisy_img = np.copy(img)
    
# # # #     # Generate salt noise (white pixels) across all channels
# # # #     num_salt = int(salt_prob * img.size // img.shape[2])  # Adjust for color channels
# # # #     coords_salt = (np.random.randint(0, img.shape[0], num_salt), 
# # # #                    np.random.randint(0, img.shape[1], num_salt))
# # # #     noisy_img[coords_salt] = [255, 255, 255]  # Set to white in RGB
    
# # # #     # Generate pepper noise (black pixels) across all channels
# # # #     num_pepper = int(pepper_prob * img.size // img.shape[2])
# # # #     coords_pepper = (np.random.randint(0, img.shape[0], num_pepper), 
# # # #                      np.random.randint(0, img.shape[1], num_pepper))
# # # #     noisy_img[coords_pepper] = [0, 0, 0]  # Set to black in RGB
    
# # # #     return noisy_img

# # # # # Add noise to the image
# # # # noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05)

# # # # # Apply Gaussian Blur
# # # # gaussian_blur = cv2.GaussianBlur(noisy_image, (5, 5), 0)

# # # # # Create a figure to display images
# # # # fig, axes = plt.subplots(1, 3, figsize=(18, 8))
# # # # fig.suptitle('Noise Removal Techniques Comparison (Color)')
# # # # axes[0].imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
# # # # axes[0].set_title('Noisy Image')
# # # # axes[1].imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
# # # # axes[1].set_title('Gaussian Blur')

# # # # # You need to define median_blur before using it
# # # # median_blur = cv2.medianBlur(noisy_image, 5)
# # # # axes[2].imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))
# # # # axes[2].set_title('Median Filtering')



# # # # for ax in axes:
# # # #     ax.axis('off')

# # # # plt.show()

# # # #####################################################
# # # import cv2
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from tensorflow.keras.models import Sequential
# # # from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Conv2DTranspose
# # # from tensorflow.keras.optimizers import Adam
# # # from sklearn.model_selection import train_test_split

# # # # Create synthetic dataset
# # # def add_salt_and_pepper_noise(img, salt_prob, pepper_prob):
# # #     noisy_img = np.copy(img)
# # #     num_salt = int(salt_prob * img.size // img.shape[2])
# # #     coords_salt = (np.random.randint(0, img.shape[0], num_salt), 
# # #                    np.random.randint(0, img.shape[1], num_salt))
# # #     noisy_img[coords_salt] = [255, 255, 255]
# # #     num_pepper = int(pepper_prob * img.size // img.shape[2])
# # #     coords_pepper = (np.random.randint(0, img.shape[0], num_pepper), 
# # #                      np.random.randint(0, img.shape[1], num_pepper))
# # #     noisy_img[coords_pepper] = [0, 0, 0]
# # #     return noisy_img

# # # # Load and preprocess a single image
# # # def preprocess_image(image_path, input_size):
# # #     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # #     image_resized = cv2.resize(image, input_size)
# # #     image_normalized = image_resized / 255.0
# # #     noisy_image = add_salt_and_pepper_noise(image_resized, salt_prob=0.05, pepper_prob=0.05) / 255.0
# # #     return noisy_image, image_normalized

# # # # Load data
# # # input_size = (128, 128)
# # # image_path = r'C:\Users\Lenovo\Downloads\blurred_images.jpg'
# # # noisy_image, clean_image = preprocess_image(image_path, input_size)

# # # # Create synthetic dataset by duplicating the single image
# # # X = np.array([noisy_image for _ in range(100)])  # Noisy inputs
# # # y = np.array([clean_image for _ in range(100)])  # Clean targets

# # # # Train-test split
# # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # # Build a simple CNN model
# # # def build_cnn(input_shape):
# # #     model = Sequential([
# # #         Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
# # #         BatchNormalization(),
# # #         ReLU(),
# # #         Conv2D(64, (3, 3), padding="same"),
# # #         BatchNormalization(),
# # #         ReLU(),
# # #         Conv2D(3, (3, 3), padding="same", activation="sigmoid")  # Output layer
# # #     ])
# # #     model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
# # #     return model

# # # # Initialize the model
# # # cnn_model = build_cnn((input_size[0], input_size[1], 3))

# # # # Train the model
# # # history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# # # # Predict using the trained model
# # # predicted_image = cnn_model.predict(np.expand_dims(noisy_image, axis=0))[0]

# # # # Visualize the results
# # # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# # # axes[0].imshow(noisy_image)
# # # axes[0].set_title("Noisy Image")
# # # axes[1].imshow(predicted_image)
# # # axes[1].set_title("CNN Output (Deblurred)")
# # # axes[2].imshow(clean_image)
# # # axes[2].set_title("Ground Truth (Clean Image)")

# # # for ax in axes:
# # #     ax.axis("off")

# # # plt.tight_layout()
# # # plt.show()
# # #####################################################################

# # import cv2
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU
# # from tensorflow.keras.optimizers import Adam
# # from sklearn.model_selection import train_test_split

# # # Create synthetic dataset
# # def add_salt_and_pepper_noise(img, salt_prob, pepper_prob):
# #     noisy_img = np.copy(img)
# #     num_salt = int(salt_prob * img.size // img.shape[2])
# #     coords_salt = (np.random.randint(0, img.shape[0], num_salt), 
# #                    np.random.randint(0, img.shape[1], num_salt))
# #     noisy_img[coords_salt] = [255, 255, 255]
# #     num_pepper = int(pepper_prob * img.size // img.shape[2])
# #     coords_pepper = (np.random.randint(0, img.shape[0], num_pepper), 
# #                      np.random.randint(0, img.shape[1], num_pepper))
# #     noisy_img[coords_pepper] = [0, 0, 0]
# #     return noisy_img

# # # Apply Gaussian filter to reduce noise
# # def apply_gaussian_filter(img, kernel_size=(5, 5), sigma=1):
# #     return cv2.GaussianBlur(img, kernel_size, sigma)

# # # Load and preprocess a single image
# # def preprocess_image(image_path, input_size):
# #     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
# #     image_resized = cv2.resize(image, input_size)
# #     image_normalized = image_resized / 255.0
# #     noisy_image = add_salt_and_pepper_noise(image_resized, salt_prob=0.05, pepper_prob=0.05) / 255.0
# #     # Apply Gaussian filter to the noisy image
# #     noisy_image_filtered = apply_gaussian_filter(noisy_image * 255).astype(np.float32) / 255.0
# #     return noisy_image_filtered, image_normalized

# # # Load data
# # input_size = (128, 128)
# # image_path = r'C:\Users\Lenovo\Downloads\blurred_images.jpg'
# # noisy_image, clean_image = preprocess_image(image_path, input_size)

# # # Create synthetic dataset by duplicating the single image
# # X = np.array([noisy_image for _ in range(100)])  # Noisy inputs
# # y = np.array([clean_image for _ in range(100)])  # Clean targets

# # # Train-test split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Build a simple CNN model
# # def build_cnn(input_shape):
# #     model = Sequential([
# #         Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
# #         BatchNormalization(),
# #         ReLU(),
# #         Conv2D(64, (3, 3), padding="same"),
# #         BatchNormalization(),
# #         ReLU(),
# #         Conv2D(3, (3, 3), padding="same", activation="sigmoid")  # Output layer
# #     ])
# #     model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
# #     return model

# # # Initialize the model
# # cnn_model = build_cnn((input_size[0], input_size[1], 3))

# # # Train the model
# # history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# # # Predict using the trained model
# # predicted_image = cnn_model.predict(np.expand_dims(noisy_image, axis=0))[0]

# # # Visualize the results
# # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# # axes[0].imshow(noisy_image)
# # axes[0].set_title("Noisy Image")
# # axes[1].imshow(predicted_image)
# # axes[1].set_title("CNN Output (Deblurred)")
# # axes[2].imshow(clean_image)
# # axes[2].set_title("Ground Truth (Clean Image)")

# # for ax in axes:
# #     ax.axis("off")

# # plt.tight_layout()
# # plt.show()
############################################3
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
# from tensorflow.keras.optimizers import Adam

# # Load the image
# image = cv2.imread(r'C:\Users\Lenovo\Downloads\blurred_images.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# # Define a function to create the U-Net model
# def unet_model(input_size=(128, 128, 3)):
#     inputs = Input(input_size)
    
#     # Encoder
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     # Bottleneck
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

#     # Decoder
#     up5 = UpSampling2D(size=(2, 2))(conv4)
#     up5 = concatenate([up5, conv3])
#     conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up5)
#     conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

#     up6 = UpSampling2D(size=(2, 2))(conv5)
#     up6 = concatenate([up6, conv2])
#     conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

#     up7 = UpSampling2D(size=(2, 2))(conv6)
#     up7 = concatenate([up7, conv1])
#     conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

#     outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv7)

#     model = Model(inputs=[inputs], outputs=[outputs])
#     return model

# # Prepare the image for the model
# def preprocess_image(image, target_size=(128, 128)):
#     image_resized = cv2.resize(image, target_size)
#     image_normalized = image_resized / 255.0
#     return image_normalized
# print ('$$$$$$$$$$$$$$$$$44')
# # Prepare the dataset (for demonstration, we will use the same image)
# input_image = preprocess_image(image)
# print ('ttttttttttttttttttttttttttttt')
# X = np.array([input_image])  # Noisy inputs
# y = np.array([input_image])  # Clean targets (for demonstration
# # Initialize the U-Net model
# model = unet_model(input_size=(128, 128, 3))
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# # Train the model (for demonstration, we will use the same image for both input and target)
# history = model.fit(X, y, epochs=10, batch_size=1)

# # Predict using the trained model
# predicted_image = model.predict(X)[0]

# # Visualize the results
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.title("Original Image")
# plt.imshow(image)
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.title("Input Image (Blurred)")
# plt.imshow(input_image)
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.title("Deblurred Image (U-Net Output)")
# plt.imshow(predicted_image)
# plt.axis('off')

# plt.show()