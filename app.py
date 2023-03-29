from flask import Flask, render_template, request
#import MusiClassifier
import tensorflow as tf
# from tensorflow.keras.models import load_model
import librosa
import utilities
from glob import glob

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    # model  = tf.keras.models.load_model('saved_model')
    # audio_file = glob("*.wav")
    # y, sr = librosa.load(audio_file[0])
    # print(audio_file)
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def data():
    # model  = tf.keras.models.load_model('saved_model')
    # audio_file = glob("*.wav")
    # prediction = model.predict("jazzmatazz_test.wav")
    genre = "Some genre" #MusiClassifier.predictFromAudio(data)
    #print(type(request.data))
    return genre

# model  = tf.keras.models.load_model('saved_model')
# audio_file = glob("*.wav")
# print(audio_file)
# prediction = model.predict("jazzmatazz_test.wav")

if __name__ == "__main__":
    app.run()
