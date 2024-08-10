import cv2
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random
import tkinter as tk
from tkinter import Label, OptionMenu, StringVar
from PIL import Image, ImageTk
import webbrowser

# Load the pre-trained models
face_classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
classifier = load_model(r"model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Set up Spotify credentials
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')

# Authenticate with Spotify using Client Credentials Flow
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=CLIENT_ID,
                                                                         client_secret=CLIENT_SECRET))

# Map moods to search queries
mood_to_query = {
    'Angry': 'angry',
    'Disgust': 'disgusting',
    'Fear': 'scary',
    'Happy': 'happy',
    'Neutral': 'neutral',
    'Sad': 'sad',
    'Surprise': 'surprise'
}

# Map language choices to query additions
language_to_query = {
    'English': '',
    'Telugu': ' telugu',
    'Tamil': ' tamil',
    'Hindi': ' hindi',
    'Malayalam': ' malayalam'
}

def get_song_by_mood_and_language(mood, language):
    query = mood_to_query.get(mood, 'pop') + language_to_query.get(language, '')
    results = sp.search(q=query, type='track', limit=25)  # Get multiple tracks
    if results['tracks']['items']:
        track = random.choice(results['tracks']['items'])  # Randomly select a track
        return track['name'], track['artists'][0]['name'], track['external_urls']['spotify']
    else:
        return None

def update_frame():
    global last_update_time, label, song_info
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    current_time = time.time()
    if current_time - last_update_time >= 5:  # Update emotion and song every 5 seconds
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Predict emotion
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                song_info = get_song_by_mood_and_language(label, selected_language.get())
                last_update_time = current_time  # Update the last update time

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label_position = (x, y)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Convert the frame to a format suitable for Tkinter
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    # Update song recommendation
    if song_info:
        song_name, artist_name, spotify_url = song_info
        song_text = f"{label} song: {song_name} by {artist_name}"
        song_label.config(text=song_text)
        link_label.config(text=spotify_url, fg="blue", cursor="hand2")
        link_label.bind("<Button-1>", lambda e: webbrowser.open_new(spotify_url))

    lmain.after(10, update_frame)

def start_main_app():
    global root
    # Destroy the language selection window
    root.destroy()
    
    # Create and configure the main application window
    global cap, lmain, song_label, link_label, last_update_time, label, song_info
    root = tk.Tk()
    root.title("Emotion Detector with Spotify Recommendations")

    # Create and place the labels and video panel
    lmain = Label(root)
    lmain.pack()

    song_label = Label(root, text="", font=('Helvetica', 12))
    song_label.pack()

    link_label = Label(root, text="", font=('Helvetica', 12), fg="blue", cursor="hand2")
    link_label.pack()

    # Initialize variables
    last_update_time = time.time()
    label = 'Neutral'  # Default label
    song_info = None

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Start the video loop
    update_frame()
    root.mainloop()

# Create Tkinter root window for language selection
root = tk.Tk()
root.title("Select Language")

selected_language = StringVar(value="English")
language_options = ["English", "Telugu", "Tamil", "Hindi", "Malayalam"]

tk.Label(root, text="Select your preferred language:", font=('Helvetica', 12)).pack(pady=10)
tk.OptionMenu(root, selected_language, *language_options).pack(pady=10)

tk.Button(root, text="Start", command=start_main_app).pack(pady=20)

# Run the language selection window
root.mainloop()
