
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
from tkinter import Label
from PIL import Image, ImageTk
import webbrowser

# Load the pre-trained models
# face_classifier = cv2.CascadeClassifier(r"Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml")
# classifier = load_model(r"Emotion_Detection_CNN-main\model.h5")

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

def get_song_by_mood(mood):
    query = mood_to_query.get(mood, 'pop')  # Default to pop if mood not found
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
                song_info = get_song_by_mood(label)
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

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create Tkinter window
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

# Start the video loop
update_frame()
root.mainloop()

# Release the capture
cap.release()
cv2.destroyAllWindows()
