import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Load model
model = pickle.load(open("digit_model.pkl", "rb"))

st.title("Handwritten Digit Recognition ")

uploaded_file = st.file_uploader("Upload a digit image (28x28 or bigger)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((8, 8))  # scikit-learn digits dataset size
    
    st.image(img, caption="Uploaded Image", width=150)

    # Preprocess
    img_array = np.array(img)
    img_array = (16 - (img_array / 16)).reshape(1, -1)

    # Predict
    pred = model.predict(img_array)
    prob = max(model.predict_proba(img_array)[0])

    st.subheader(f"Predicted Digit: {pred[0]}")
    st.write(f"Confidence: {prob * 100:.2f}%")
