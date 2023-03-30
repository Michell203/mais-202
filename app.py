import wave
from flask import Flask, render_template, request
import keras
import numpy as np
import tensorflow as tf
import librosa
from pydub import AudioSegment
import math

app = Flask(__name__)
#model should probably be an attribute

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def data():
    model = tf.keras.models.load_model('./saved_model')
    songFeatures = transform(request.data)
    genre = predict(model, songFeatures)
    print(genre)
    return {"genre":genre}

def transform(audioFile):
    # Load the audio file
    audio, sr = librosa.load(audioFile, sr=None)

    # Set the segment length and hop length in seconds
    segment_length = 3
    hop_length = 512
    print(sr)
    # Calculate the number of samples in a segment
    segment_samples = int(segment_length * sr)

    # Segment the audio
    segments = []
    for i in range(0, len(audio), hop_length):
        segment = audio[i:i+segment_samples]
        if len(segment) == segment_samples:
            segments.append(segment)

    # Extract audio features for each segment
    features = []
    print(len(segments))
    for segment in segments:
        # Extract Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=segment, sr=sr,n_mfcc=13,n_fft=2048,hop_length=512)
        # Extract spectral centroid
        cent = librosa.feature.spectral_centroid(y=segment, sr=sr)
        # Extract zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=segment)
        # Concatenate all features
        feature_vector = np.concatenate(mfccs.flatten(), cent.flatten(), zcr.flatten())
        features.append(feature_vector)

    # Convert the feature list to a numpy array
    features = np.array(features)
    return features

def predict(model,X):

    X=X[np.newaxis,...]
    
    prediction=model.predict(X)


    #get index with max value
    predictIndex=np.argmax(prediction, axis=1)

    print(predictIndex)
    genres = ["Blues", "Classical", "Country", "Disco", "Hip-Hop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
    return genres[predictIndex]

def createModel():
    model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(58,)),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer = 'adam',
                loss='sparse_categorical_crossentropy',
                metrics='accuracy'
                )
    
    fname = "model/weights.hdf5"
    model.load_weights(fname)
    
    return model


audio = "../Her Majesty (Remastered 2009).wav"
X = transform(audio)
print(X.shape)
model = tf.keras.models.load_model('./saved_model')
print(predict(model, X))
