import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO

# Set Streamlit page config
st.set_page_config(page_title="Facial Gender Classifier", page_icon="🧠", layout="centered")

# Load the trained model
model = load_model("gender_classifier_model.h5")

# Sidebar info
with st.sidebar:
    st.title("🧠 Gender Classifier")
    st.markdown("Upload or capture a face image, and the model will predict the **gender** (Male or Female).")
    st.markdown("---")
    st.markdown("🔹 Model: CNN\n🔹 Input size: 128x128\n🔹 Output: Male / Female")
    st.markdown("---")
    st.caption("Developed by  Nithin Punyamanthula ")

# Main app content
st.header("👤 Facial Gender Detection")
st.write("Upload an image or use your **webcam** to detect gender.")

# Inputs: Webcam or Upload
webcam_image = st.camera_input("📷 Capture an image")
uploaded_file = st.file_uploader("📁 Or upload an image", type=["png", "jpg", "jpeg"])

# Use image if available
if webcam_image or uploaded_file:
    img_source = BytesIO(webcam_image.getvalue()) if webcam_image else uploaded_file
    img = Image.open(img_source).convert("RGB")  # Ensure RGB format

    st.image(img, caption="📸 Input Image", use_column_width=True)

    # Preprocess
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Predicting gender..."):
        prediction = model.predict(img_array)[0][0]
        gender = "👩 Female" if prediction < 0.5 else "👨 Male"

    st.success(f"**Predicted Gender:** {gender}")

else:
    st.info("Please capture or upload a facial image to begin.")

