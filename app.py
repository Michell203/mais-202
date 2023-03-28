from flask import Flask, render_template, request
import keras
#import MusiClassifier
#import tensorflow as tf

app = Flask(__name__)
#model should probably be an attribute

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def data():
    model = createModel()
    song = transform(request.data)
    genre = predict(song, model)
    print(genre)
    return {"genre":genre}

def transform(audioFile):
    #TODO:
    # Transform the .wav file into unsable array of features
    return None

def predict(song):
    #TODO:
    # use the model attribute and return the genre
    return None

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
