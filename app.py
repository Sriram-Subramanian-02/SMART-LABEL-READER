import streamlit as st
import os
from google_search.services import rag_pipeline
from image_extractor.services import get_ingredients

# Sidebar input elements
st.sidebar.header("Input Information")

# Age
age = st.sidebar.number_input('Age', min_value=18, max_value=80, value=22)

# Gender
gender = st.sidebar.radio("Gender", ('Male', 'Female'))

# Height
height = st.sidebar.number_input('Height (cm)', value=185)

# Weight
weight = st.sidebar.number_input('Weight (kg)', value=65)

# Activity level dropdown
activity = st.sidebar.selectbox('Activity', 
                                ['Basal Metabolic Rate (BMR)', 
                                 'Sedentary: little or no exercise', 
                                 'Light: exercise 1-3 times/week', 
                                 'Moderate: exercise 4-5 times/week', 
                                 'Active: daily exercise or intense exercise 3-4 times/week', 
                                 'Very Active: intense exercise 6-7 times/week', 
                                 'Extra Active: very intense exercise daily, or physical job'])

# Image uploader
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Chat input
user_input = st.chat_input("Say something")

# Define the path to save the image
save_path = "F:\\psg\\sem_9\\ir\\package\\user_input_data"
file_name = None

if user_input and uploaded_image:
    # Display user input and image in chat
    with st.chat_message("user"):
        st.write(f"{user_input}")
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image to the specified path
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # Create the directory if it doesn't exist

        # Create the file path for saving the image
        file_name = os.path.join(save_path, uploaded_image.name)
        
        # Save the uploaded image
        with open(file_name, "wb") as f:
            f.write(uploaded_image.getbuffer())

    with st.chat_message("ai"):
        ingredients = get_ingredients(file_name)
        response = rag_pipeline(user_input, "healthy snack", ingredients, age, gender, height, weight, activity)
        st.write(f"{response}")
