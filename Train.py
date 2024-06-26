import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import random
def rotate_randomly(image, steering_angle):
    # Randomly choose the angle of rotation between -90 and 90 degrees
    angle = random.randint(-25, 25)
    steering_angle = steering_angle-angle

    # Get the center of the image
    center = (image.shape[1] // 2, image.shape[0] // 2)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return rotated_image,steering_angle


def add_random_shadow(image):
    h, w = image.shape[:2]
    top_x, top_y = w * np.random.uniform(), 0
    bot_x, bot_y = w * np.random.uniform(), h
    shadow_mask = np.zeros_like(image)

    polygon = np.array([[top_x, top_y], [bot_x, bot_y], [w, bot_y], [w, top_y]], dtype=np.int32)
    cv2.fillPoly(shadow_mask, [polygon], 255)

    shadow_ratio = np.random.uniform(0.5, 0.75)
    cond1 = shadow_mask[:, :, 0] == 255
    image[cond1] = image[cond1] * shadow_ratio

    return image
def read_and_preprocess_image(file_path, steering_angle):
    # Read grayscale image
    image = cv2.imread(file_path)

    if image is not None:  # Check if image is not None
        # Data augmentation
        if random.choice([True, False]):
            # Randomly adjust brightness
            random_brightness = 0.25 + np.random.uniform()
            image = image * random_brightness
            # Clip values to ensure they are in the valid range [0, 255]
            image = np.clip(image, 0, 255)

        if random.choice([True, False]):
            image, steering_angle = rotate_randomly(image, steering_angle)

        if random.choice([True, False]):
            # Apply random noise
            noise = np.random.normal(loc=0, scale=25, size=image.shape)
            image = image + noise
            # Clip values to ensure they are in the valid range [0, 255]
            image = np.clip(image, 0, 255)
        if random.choice([True, False]):
            image = add_random_shadow(image)

        # Normalize pixel values to [0, 1]
        image = image / 255.0
        # Add a singleton dimension to make it (height, width, 1)
        image = np.expand_dims(image, axis=0)

    return image, steering_angle

def batch_generator(image_paths, steering_angles, batch_size):
    num_samples = len(image_paths)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_images = []
            batch_angles_list = []  # Use a different name to store angles as a list
            batch_paths = image_paths[offset:offset + batch_size]
            batch_angles = steering_angles[offset:offset + batch_size]
            for i in range(len(batch_paths)):
                # For training set
                image, angle = read_and_preprocess_image(batch_paths[i], batch_angles[i])
                batch_images.append(image)
                batch_angles_list.append([angle])
            yield np.concatenate(batch_images), np.array(batch_angles_list)
def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation="elu"))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))
    model.add(Flatten())
    model.add(Dense(1164, activation="elu")) #1164
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.0001), loss="mse", run_eagerly=True)
    return model

# Load image paths and steering angles from txt file
data_folder = r"C:\Users\arsha\PycharmProjects\CPU\OLD\dataset\img"
txt_file_path = r"C:\Users\arsha\PycharmProjects\CPU\OLD\dataset\modified_steering.txt"

image_paths = []
steering_angles = []

with open(txt_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        parts = line.split()
        image_paths.append(os.path.join(data_folder, parts[0]))
        steering_angles.append(float(parts[1]))

# Create data generators
batch_size = 24
train_generator = batch_generator(image_paths, steering_angles, batch_size)

# Create an instance of the model
model = nvidia_model()
model.summary()
# Split your data into training and validation sets
from sklearn.model_selection import train_test_split

# Assuming you have image_paths and steering_angles from before
X_train, X_val, y_train, y_val = train_test_split(image_paths, steering_angles, test_size=0.2, random_state=42)

# Create data generators for both training and validation
train_generator = batch_generator(X_train, y_train, batch_size)
val_generator = batch_generator(X_val, y_val, batch_size) 

# Train the model with validation data
history = model.fit(train_generator, steps_per_epoch=len(X_train)//batch_size, epochs=10,
                    validation_data=val_generator, validation_steps=len(X_val)//batch_size)


# Save the model
model.save('model/MK3.h5')