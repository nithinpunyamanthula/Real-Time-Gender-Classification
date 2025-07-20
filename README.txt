# 👤 Facial Gender Detection using CNN 🧠

A deep learning-based project that detects **gender (Male or Female)** from facial images using a Convolutional Neural Network (CNN), with a real-time web interface built in **Streamlit**.

---


# 🔍 Overview

This project implements a gender classification system using deep learning. It is capable of predicting gender (Male/Female) from a facial image, using a CNN trained on the UTKFace dataset. Users can interact with the model through a simple Streamlit web application that allows image upload or webcam capture.


---

# 🧠 Model Architecture

- **Input:** 128x128 RGB facial image
- **Layers:**
  - 3 Convolutional layers (Conv2D + MaxPooling)
  - Flatten layer
  - Dense layers with Dropout
  - Output layer with Sigmoid activation
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

---

# 📂 Dataset

- **Source:** [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- **Images:** 20,000+ labeled images
- **Labels:** Extracted from filenames (format: age_gender_race_date.jpg)
  - `gender` is at index 1: `0 = Male`, `1 = Female`
- **Preprocessing:**
  - Resized to 128x128
  - Normalized pixel values
  - Split into train, validation, and test sets

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/nithinpunyamanthula/Real-Time-Gender-Classification
cd facial-gender-detection

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
