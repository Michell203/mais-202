import wave
from flask import Flask, render_template, request
# import keras
import numpy as np
import tensorflow as tf
import librosa
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
#model should probably be an attribute

reconstructed_model = tf.keras.models.load_model("my_model_h5.h5", compile=False)
reconstructed_model.compile(optimizer = "adam",
                loss='sparse_categorical_crossentropy',
                metrics='accuracy')

model_features = ["chroma_stft_mean","chroma_stft_var","rms_mean","rms_var","spectral_centroid_mean","spectral_centroid_var","spectral_bandwidth_mean","spectral_bandwidth_var","rolloff_mean","rolloff_var","zero_crossing_rate_mean","zero_crossing_rate_var","harmony_mean","harmony_var","perceptr_mean","perceptr_var","tempo","mfcc1_mean","mfcc1_var","mfcc2_mean","mfcc2_var","mfcc3_mean","mfcc3_var","mfcc4_mean","mfcc4_var","mfcc5_mean","mfcc5_var","mfcc6_mean","mfcc6_var","mfcc7_mean","mfcc7_var","mfcc8_mean","mfcc8_var","mfcc9_mean","mfcc9_var","mfcc10_mean","mfcc10_var","mfcc11_mean","mfcc11_var","mfcc12_mean","mfcc12_var","mfcc13_mean","mfcc13_var","mfcc14_mean","mfcc14_var","mfcc15_mean","mfcc15_var","mfcc16_mean","mfcc16_var","mfcc17_mean","mfcc17_var","mfcc18_mean","mfcc18_var","mfcc19_mean","mfcc19_var","mfcc20_mean","mfcc20_var"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def data():
    songFeatures = request.data
    file = request.files['file']
    print(songFeatures.filename)
    print("songFeatures: \n")
    # print(songFeatures)
    genre = predict(songFeatures)
    # print(genre)
    return {"genre":genre}

def predict(pred):

    predi = os.path.basename(os.path.normpath(pred))

    frame = pd.read_csv("Data/features_30_sec.csv")

    select_song = frame.loc[frame['filename'] == str(predi)]
    select_song = select_song.drop('filename', axis=1)

    fit = StandardScaler()

    select_song_arr = select_song.values
    select_song_arr = select_song_arr[0][:-1]
    select_song_arr = np.array(select_song_arr)
    select_song_arr = np.reshape(select_song_arr,(1,58))
    select_song_arr[0][0] = 0
    select_song_arr = np.reshape(select_song_arr,(58,1))

    X = fit.fit_transform(select_song_arr)
    X = np.reshape(X,(1,58))
    select_song_arr = np.asarray(select_song_arr).astype('float32')

    predict = reconstructed_model.predict(X)
    predictIndex = np.argmax(predict, axis=1)
    genres = ["Blues", "Classical", "Country", "Disco", "Hip-Hop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]

    print(genres[predictIndex[0]])
    return genres[predictIndex[0]]

# def createModel():
#     model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(512, activation='relu', input_shape=(58,)),
#     tf.keras.layers.Dropout(0.2),

#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.2),

#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
    
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.2),

#     tf.keras.layers.Dense(10, activation='softmax'),
#     ])

#     model.compile(optimizer = 'adam',
#                 loss='sparse_categorical_crossentropy',
#                 metrics='accuracy'
#                 )
    
#     fname = "model/weights.hdf5"
#     model.load_weights(fname)
    
#     return model

def transform(audioFile):
     # Load the audio file
     audio, sr = librosa.load(audioFile, sr=None)

     # Set the segment length and hop length in seconds
     segment_length = 15
     hop_length = 3 * sr
     # Calculate the number of samples in a segment
     segment_samples = int(segment_length * sr)

     # Segment the audio
     segments = []
     t1 = 0 
     t2 = segment_samples
     while t2 < len(audio):
         segment = audio[t1:t2]
         segments.append(segment)
         t1 = t2 - hop_length
         t2 = t1 + segment_samples
     segments.append(audio[t1:])

     print(len(segments))
     features = [[0 for i in range(58)] for j in segments]
     print("({}, {})".format(len(features), len(features[0])))
     for segment in range(len(segments)):
         features[segment][0] = len(segments[segment]) * sr

         stft = librosa.feature.chroma_stft(y=segments[segment], sr=sr, hop_length=hop_length)
         features[segment][1] = np.mean(stft)
         features[segment][2] = np.var(stft)
         rms = librosa.feature.rms(y=segments[segment], frame_length=len(segments[segment]), hop_length=hop_length)
         features[segment][3] = np.mean(rms)
         features[segment][4] = np.var(rms)

         spec_centroid = librosa.feature.spectral_centroid(y=segments[segment], sr=sr, hop_length=hop_length)
         features[segment][5] = np.mean(spec_centroid)
         features[segment][6] = np.var(spec_centroid)

         spec_bandwith = librosa.feature.spectral_bandwidth(y=segments[segment], sr=sr, hop_length=hop_length)
         features[segment][7] = np.mean(spec_bandwith)
         features[segment][8] = np.var(spec_bandwith)

         rollof = librosa.feature.spectral_rolloff(y=segments[segment], sr=sr, hop_length=hop_length)
         features[segment][9] = np.mean(rollof)
         features[segment][10] = np.var(rollof)

         zero_cross = librosa.feature.zero_crossing_rate(y=segments[segment], frame_length=len(segments[segment]), hop_length=hop_length)
         features[segment][11] = np.mean(zero_cross)
         features[segment][12] = np.var(zero_cross)
         harmony = librosa.effects.harmonic(y=segments[segment]) # WE NEED TO  LOOK AT THIS TO BE SURE IT'S THE HARMONY FEATURE
         features[segment][13] = np.mean(harmony)
         features[segment][14] = np.var(harmony)

         percept = 0.5 #librosa.feature.tonnetz(y=segments[segment], sr=sr, hop_length=hop_length)
         features[segment][15] = 0.5#np.mean(percept)
         features[segment][16] = 0.5#np.var(percept)

         tempo = librosa.feature.tempo(y=segments[segment], sr=sr, hop_length=hop_length)
         features[segment][17] = np.mean(tempo)

         mfccs = librosa.feature.mfcc(y=segments[segment], sr=sr, n_mfcc=20, hop_length=hop_length)
         for i in range(0, 20):
             features[segment][2*i+18] = np.mean(mfccs[i])
             features[segment][2*i+19] = np.var(mfccs[i])

    
     features0 = features[0]
    
     return features0

# Test run
audio = "../Her Majesty (Remastered 2009).wav"
#reconstructed_model = tf.keras.models.load_model("my_model_h5.h5", compile=False)
#reconstructed_model.compile(optimizer = "adam",
#                loss='sparse_categorical_crossentropy',
#                metrics='accuracy')
#audio = "/Her_Majesty_Remastered_2009.wav"
X = transform(audio)
model = tf.keras.models.load_model('./saved_model',compile=False)
model.compile(optimizer = "adam",
                loss='sparse_categorical_crossentropy',
                metrics='accuracy')
# print(predict(reconstructed_model, X[0]))
# print(X)
# reconstructed_model.predict(X)
Xn = np.array(X)
print(Xn.shape)
Xn = np.tile(Xn, (128, 1))
predict = model.predict(Xn)
print(predict[0])
predictIndex = np.argmax(predict, axis=1)
# pred = predict('Data/genres_original/blues/blues.00000.wav')


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host=os.getenv('IP', '192.168.0.55'), 
            # port=int(os.getenv('PORT', 4445)), debug=True)