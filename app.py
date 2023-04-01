import wave
from flask import Flask, render_template, request
# import keras
import numpy as np
import tensorflow as tf
import os
import librosa
import pandas as pd
# from pydub import AudioSegment
import math

app = Flask(__name__)
#model should probably be an attribute

# reconstructed_model = tf.keras.models.load_model("my_model_h5.h5")
model_features = ["chroma_stft_mean","chroma_stft_var","rms_mean","rms_var","spectral_centroid_mean","spectral_centroid_var","spectral_bandwidth_mean","spectral_bandwidth_var","rolloff_mean","rolloff_var","zero_crossing_rate_mean","zero_crossing_rate_var","harmony_mean","harmony_var","perceptr_mean","perceptr_var","tempo","mfcc1_mean","mfcc1_var","mfcc2_mean","mfcc2_var","mfcc3_mean","mfcc3_var","mfcc4_mean","mfcc4_var","mfcc5_mean","mfcc5_var","mfcc6_mean","mfcc6_var","mfcc7_mean","mfcc7_var","mfcc8_mean","mfcc8_var","mfcc9_mean","mfcc9_var","mfcc10_mean","mfcc10_var","mfcc11_mean","mfcc11_var","mfcc12_mean","mfcc12_var","mfcc13_mean","mfcc13_var","mfcc14_mean","mfcc14_var","mfcc15_mean","mfcc15_var","mfcc16_mean","mfcc16_var","mfcc17_mean","mfcc17_var","mfcc18_mean","mfcc18_var","mfcc19_mean","mfcc19_var","mfcc20_mean","mfcc20_var"]

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
        feature_vector = np.concatenate((mfccs.flatten(), cent.flatten(), zcr.flatten()))


        features.append(feature_vector)
        animals = pd.DataFrame(features,model_features)
        animals.to_csv("file.csv")
        # features = np.array(features)
        feature_data = pd.read_csv("file.csv")
        # features = features[model_features]
        features = feature_data[model_features]

    # Convert the feature list to a numpy array
    # features = np.array(features)
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
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(58,)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer = 'adam',
                loss='sparse_categorical_crossentropy',
                metrics='accuracy'
                )
    
    fname = "model/weights.hdf5"
    model.load_weights(fname)
    
    return model

if __name__ == "__main__":
    # app.run()
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4445)))


audio = "./Her_Majesty_Remastered_2009.wav"
X = transform(audio)
print(X.shape)
model = tf.keras.models.load_model('./saved_model')
print(predict(model, X))
