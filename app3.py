import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageOps
import io
import numpy as np
import cv2

# Load the trained model
class_names = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 
               'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=20, bias=True)
model.load_state_dict(torch.load('cnn_model1.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Define a function to predict the class
def predict(image, model, transform):
    # Convert the image to RGB
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Function to segment characters from the image
def segment_characters(image):
    gray_image = image.convert('L')
    np_image = np.array(gray_image)
    _, binary_image = cv2.threshold(np_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    segmented_chars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_image = binary_image[y:y+h, x:x+w]
        char_image = Image.fromarray(char_image)
        segmented_chars.append(char_image)
    
    return segmented_chars

# Streamlit app
st.title("Aksara Jawa Detection")

# Camera input
image_data = st.camera_input("Take a picture")

if image_data is not None:
    # Load the image
    image = Image.open(io.BytesIO(image_data.getvalue()))
    
    # Display the image
    st.image(image, caption='Captured Image', use_column_width=True)
    
    # Segment characters from the image
    segmented_chars = segment_characters(image)
    
    # Predict each character
    recognized_text = ""
    for char_image in segmented_chars:
        char_class = predict(char_image, model, transform)
        recognized_text += char_class + " "
    
    # Display the recognized text
    st.write(f"Recognized Text: {recognized_text}")
