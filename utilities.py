from flask import Flask, render_template, request
#import MusiClassifier
import tensorflow as tf
# from tensorflow.keras.models import load_model
import librosa
import utilities
from glob import glob

def predictFromAudio(m, wav):
  #TODO: transform the audio file into a readable array of info, then make a prediction and return the most probable genre 

  return ("Not done yet")


# model  = tf.keras.models.load_model('saved_model')
# audio_file = glob("*.wav")
# y, sr = librosa.load(audio_file[0])
# print(audio_file)
# prediction = model.predict("jazzmatazz_test.wav")