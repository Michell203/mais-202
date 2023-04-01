from flask import Flask, render_template, request
import keras
import numpy as np
import tensorflow as tf
import librosa

app = Flask(__name__)
#model should probably be an attribute

@app.route("/")
def index():
    # model  = tf.keras.models.load_model('saved_model')
    # audio_file = glob("*.wav")
    # y, sr = librosa.load(audio_file[0])
    # print(audio_file)
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def data():
    model = tf.keras.models.load_model('./saved_model')
    songFeatures = transform(request.data)
    genre = predict(model, songFeatures)
    print(genre)
    return {"genre":genre}

def predict(model,X):
    
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
    
    return features

# Test run
audio = "../Her Majesty (Remastered 2009).wav"
X = transform(audio)
model = tf.keras.models.load_model('./saved_model')
print(predict(model, X[0]))