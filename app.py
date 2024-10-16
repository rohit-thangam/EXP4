import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('C:/SSD/BIT/S5-LAB/ML/EXP4/mnist_digit_classifier.h5')

st.title("Handwritten Digit Classification")
# Streamlit file uploader
uploaded_file = st.file_uploader("Choose a handwritten digit image...", type="png")

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize image to 28x28
    image_array = np.array(image)  # Convert to array
    image_array = image_array.astype('float32') / 255  # Normalize the image
    image_array = image_array.reshape(1, 28, 28, 1)  # Add batch dimension and channel

    # Flatten the image to match the model input
    image_array = image_array.reshape(1, 784)  # Reshape to (1, 784)

    # Make predictions
    predictions = model.predict(image_array)
    predicted_digit = np.argmax(predictions)

    st.write(f"**Predicted digit: {predicted_digit}**")
