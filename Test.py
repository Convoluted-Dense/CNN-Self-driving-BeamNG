import cv2
import mss
import numpy as np
import pyvjoy
from PIL import Image
from PIL import ImageGrab,ImageFilter
from keras.models import load_model
from pynput import keyboard


joy = pyvjoy.VJoyDevice(1)

# Function to simulate steering input
def simulate_steering(angle):
    normalized_angle = max(-1, min(1, angle / 90.0))

    axis_value = int((normalized_angle + 1) * 16383.5)

    joy.set_axis(pyvjoy.HID_USAGE_X, axis_value)


model = load_model('model/MK3.h5')

def preprocess_screen():
    with mss.mss() as sct:
        monitor_number = 2
        mon = sct.monitors[monitor_number]
        monitor = {
            "top": mon["top"],
            "left": mon["left"],
            "width": mon["width"],
            "height": mon["height"],
            "mon": monitor_number
        }

        sct_img = sct.grab(monitor)
        screenshot = np.array(sct_img)

    image = Image.fromarray(screenshot)

    # Define the cropping points
    top_left = (0, 476)
    bottom_right = (1911, 779)

    # Crop the image
    cropped_image = image.crop((*top_left, *bottom_right))
    image_blurred = cropped_image.filter(ImageFilter.BLUR)

    # Convert PIL Image to numpy array
    image_blurred = np.array(image_blurred)

    # Convert RGB to BGR (since OpenCV uses BGR format)
    image_blurred = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2BGR)

    # Convert BGR to HSV
    image_blurred = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2HSV)

    # Resize the image
    resized_image = cv2.resize(image_blurred, (200, 66))

    # Normalize pixel values to [0, 1]
    normalized_screen = np.array(resized_image) / 255.0


    # Add batch dimension
    processed_screen = np.expand_dims(normalized_screen, axis=0)

    return processed_screen

# Function to predict steering angle
def predict_steering_angle():
    processed_screen = preprocess_screen()
    processed_screen = np.expand_dims(processed_screen, axis=-1)

    predictions = model.predict([processed_screen])
    steering_angle = float(predictions[0][0])
    steering_angle =steering_angle

    return steering_angle

def on_key_press(key):
    try:
        if key.char == 'l':
            global mouse_movement_enabled
            mouse_movement_enabled = not mouse_movement_enabled
    except AttributeError:
        pass


listener = keyboard.Listener(on_press=on_key_press)
listener.start()

# Main loop for capturing the screen and making predictions
mouse_movement_enabled = True
while True:

    x = processed_screen = preprocess_screen()
    predicted_angle = predict_steering_angle()


    # Print the predicted angle and confidence
    print(f"Predicted Angle: {predicted_angle}")



    if mouse_movement_enabled:
        # Control the mouse based on the predicted angle
        simulate_steering(predicted_angle)

    cv2.imshow("Autonomous Driving Demo", cv2.resize(processed_screen[0], (200, 66)))

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the OpenCV window
cv2.destroyAllWindows()
