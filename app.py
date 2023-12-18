# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:15:57 2023

@author: Telugu Bharadwaj
"""



import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt

def generate_heat_map(img_array, model, last_conv_layer_name):
    # Create a model that maps the input image to the desired conv layer output
    conv_model = tf.keras.models.Model(model.inputs, model.get_layer(last_conv_layer_name).output)
    
    # Convert the input array to a tensor
    img_tensor = tf.convert_to_tensor(img_array)
    
    # Get the conv layer output
    conv_outputs = conv_model(img_tensor)
    
    # Multiply each channel in the conv layer output by the maximum value
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    return heatmap

def overlay_heatmap(image, heatmap):
    # Normalize the heatmap values between 0 and 255
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Resize the heatmap to match the size of the image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert the heatmap to the required format for applying the color map
    heatmap = cv2.convertScaleAbs(heatmap)

    # Apply the color map to the heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Apply the heatmap as an overlay on the image
    overlaid_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return overlaid_img





def main():
    st.title("Malaria Detector Web App")
    activity = ['Little Description', 'Prediction', 'About']
    choice = st.sidebar.selectbox('Choose an Activity', activity)

    if choice == 'Little Description':
        st.subheader("ORIGINAL DATA SOURCE")
        st.text("The dataset contains 2 folders - Infected - Uninfected")
        st.text("Acknowledgements: This Dataset is taken from the official NIH Website:")
        st.markdown("https://ceb.nlm.nih.gov/repositories/malaria-datasets/")

    if choice == 'Prediction':
        st.subheader("Malaria Infection Prediction")

        # Load the pre-trained model
        model = tf.keras.models.load_model("model.h5")

        # Display instructions
        st.write("Please select an image for prediction:")

        # Image selection
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the selected image
            img = image.load_img(uploaded_file, target_size=(68, 68))
            st.image(img, caption='Selected Image', use_column_width=True)

            # Preprocess the image
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Perform prediction
            prediction = model.predict(img_array)

            if prediction[0][0] > 0.5:
                st.write("Prediction: Uninfected")
                st.write("Probability: {:.2%}".format(prediction[0][0]))
            else:
                st.write("Prediction: Infected/Parasitized")
                st.write("Probability: {:.2%}".format(prediction[0][0]))

            # Generate heat map
            last_conv_layer_name = 'conv2d'
            heatmap = generate_heat_map(img_array, model, last_conv_layer_name)

            # Overlay heatmap on the original image
            overlaid_img = overlay_heatmap(img_array[0], heatmap)

            # Display the overlaid image
            st.image(overlaid_img, caption='Heatmap Overlay', use_column_width=True)

    if choice == 'About':
        st.subheader("Malaria Detection Web App made with Streamlit by Bharadwaj")
        st.info("Email: telugu.bharadwaj@gmail.com")


if __name__ == '__main__':
    main()
