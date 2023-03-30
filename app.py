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
    #TODO:
    # Transform the .wav file into unsable array of features
    segmentedList = segment(audioFile)
    print(segmentedList.size)

    return None

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

def segment(audiofile):
    segmentduration = 3000 #milliseconds
    segmentedList = np.array([])
    audio = AudioSegment.from_wav(audiofile)
    duration = math.floor(audio.duration_seconds * 1000)
    print(duration)
    t1 = 0
    t2 = t1 + segmentduration
    print(audio[t1:t2])
    while t2 < duration:
        segmentedList = np.append(segmentedList, segment)
        print(segmentedList)
        t1 = t1 + segmentduration
        t2 = t2 + segmentduration
    
    segmentedList = np.append(segmentedList, audio[t1:duration])

    return segmentedList


audio = "../Her Majesty (Remastered 2009).wav"
transform(audio)