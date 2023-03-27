from flask import Flask, render_template, request
#import MusiClassifier
import tensorflow as tf

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def data():
    model  = tf.keras.models.load_model('./saved_model')
    genre = "Some genre"#MusiClassifier.predictFromAudio(data)
    #print(type(request.data))
    return genre

if __name__ == "__main__":
    app.run()
