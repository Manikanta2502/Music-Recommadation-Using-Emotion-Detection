# Emotion-Based Music Recommendation System

## Data Set Link : [Face Expression Recognition Dataset](https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset)

This project is an advanced emotion-based music recommendation system that detects user emotions and provides tailored music recommendations. It integrates with Spotify and supports multiple languages, including English, Telugu, Tamil, Hindi, and Malayalam.

## Features Overview

- **Emotion Detection**: Detects user emotions using a pre-trained model in Keras.
- **Music Recommendation**: Provides personalized song recommendations based on the detected emotion, leveraging Spotify's API.
- **Multi-Language Support**: Recommends songs in multiple languages, including English, Telugu, Tamil, Hindi, and Malayalam.

## Prerequisites

- **Python**: Ensure you have Python installed. You can download it [here](https://www.python.org/downloads/).
- **Spotify Developer Account**: You need to set up a Spotify Developer account and obtain the necessary API keys. Sign up [here](https://developer.spotify.com/).
- **Pre-trained Model**: A pre-trained emotion detection model is required. You can download a model or train your own.
- **Camera Access**: Ensure you have a camera-access device connected to run the emotion detection feature.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Manikanta2502/Music-Recommendation-Using-Emotion-Detection.git
   cd Music-Recommendation-Using-Emotion-Detection
Here's the bash code from the README file:

1. **(Optional) Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Environment Variables**:
   Create a `.env` file in the root directory and add your Spotify API keys:
   ```bash
   SPOTIFY_CLIENT_ID=your_spotify_client_id_here
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
   ```

4. **Running the Project**:
   ```bash
   python src/main.py
   ```
5. **Check File Locations:**
Ensure that the haarcascade_frontalface_default.xml file and the model.h5 file are in the correct locations as specified in your project setup.

Verify that you are working in the correct terminal environment.
