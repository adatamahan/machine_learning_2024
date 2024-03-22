

import numpy as np
import streamlit as st
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import invert
from PIL import Image
import joblib


# Streamlit app
def main():
    st.title("Digit Recognition App")
    
    # File uploader widget
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Read the uploaded image
        image = np.array(Image.open(uploaded_image))
        # Convert the resized image to grayscale
        image_gray = rgb2gray(image)
        # invert image
        image_invert = invert(image_gray)
        # Resize the image to 28x28 pixels
        image_resized = resize(image_invert, (28, 28))
        # scale pixels to 0-255
        scaled_image = (image_resized * 255).astype(np.uint8)
        # create a binary image
        binary_image = np.where(scaled_image < 125, 0, 255)
        # Flatten the image into a 1D array
        img_array = binary_image.flatten()   
        
        # Create a layout for displaying images horizontally
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(image, caption='Original Image', width=100)
        with col2:
            st.image(image_gray, caption='Gray Image', width=100)
        with col3:
            st.image(image_invert, caption='Inverted Image', width=100)
        with col4:
            st.image(image_resized, caption='Resized Image', width=100)
        with col5:
            st.image(binary_image, caption='Binary Image', width=100) 
        
        # Prediction
        if st.button('Prediction'):
            model = joblib.load("final_model.pkl")
            prediction = model.predict(img_array.reshape(1, -1))
            st.markdown("# Predicted number: " + str(prediction))
         
if __name__ == "__main__":
    main()

     
           
      