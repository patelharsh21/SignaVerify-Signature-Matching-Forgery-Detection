

# Signature Verification Web App

## Overview

This web application verifies if two physical pen signatures belong to the same person using a deep learning model built with TensorFlow. The core functionality involves a Siamese neural network that processes both signature images, calculates the Euclidean distance between their latent vectors, and determines their similarity.
## Video Demo 

[LINK](https://drive.google.com/drive/folders/1lpf7UnG-tjJbyZE9RelnPfa4YVH1CUdD)
## High level Diagram 

![image](https://github.com/user-attachments/assets/25750958-3ce4-47ba-96e2-d578a92066d9)

## Tech Stack

- **Flask:** Lightweight and flexible Python web framework used for building the backend.
- **TensorFlow:** Utilized for building and deploying the Siamese neural network model.
- **HTML/CSS/JavaScript:** Frontend technologies used to create a responsive and user-friendly interface.

## How It Works

1. **Image Upload:** The user uploads two signature images.
2. **Preprocessing:** The images are preprocessed to prepare them for model input.
3. **Siamese Network:** Both images are passed through a Siamese neural network.
4. **Euclidean Distance Calculation:** The latent vectors from the network are compared using Euclidean distance.
5. **Result:** The app displays whether the signatures match (same person) or not.
## Low level diagram

![image](https://github.com/user-attachments/assets/0c061f9c-0a5b-40d8-b0dc-9859ebf1ff82)

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git git@github.com:patelharsh21/SignaVerify-Signature-Matching-Forgery-Detection.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd SignaVerify-Signature-Matching-Forgery-Detection
   ```
3. **Create and Activate a Virtual Environment:**
   ```bash
   conda create --name sign_forgery python==3.12.4
   conda activate sign_forgery
   ```
4. **Install the Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the Flask App:**
   ```bash
   python app.py
   ```
6. **Access the App in Your Browser:**
   Open `http://127.0.0.1:5000` in your web browser.



