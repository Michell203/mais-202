# MusiClassifier
Project repository for MAIS 202. The project is a music genre classifier using [this](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset from Kaggle. We will be using CNN Algorithm to train our model. The model is based on [this](https://www.kaggle.com/code/satoru90/music-genre-classification-xgb-deep-learning/notebook) submission on kaggle with some tweaks, but it still needs some improvement.

The webapp can identify the genre of a song using a CNN model based on a number of features that go from the tempo to the mel spectrogram. It can identify the following genres with an accuracy of around 93%:
* Blues
* Classical
* Country
* Disco
* Hip Hop
* Jazz
* Metal
* Pop
* Raggae
*Rock

## How to use it?

Submit a .wav of a song in the appropriate box and click on the classify button. The song's genre should appear after a few seconds

## TODO
We still have many imporvements to make on this app...

* Implement the method that will allow to make a prediction from an audio file. That includes verifying if the file is actually an audiofile and handeling the case where it's not, the feature extraction and prediction from the model implemented with previously saved weights.
* The model can be upgraded. It is slightly overfitted to the training data. Early stopping is probably best as the training is set to a constant number of training runs instead of stopping when the accuracy stops improving. Some Hyperparameters could be improved or, at least, some more testing needs to be done to compare the results for different hyperparameters
* Make the web page look nicer. Ideas:
  * Display an image representing the identified genre
  * Add colour
  * Center the components
  * Actually showing the melSpectrograms and frequency plot could be a nice addition
* Implement a way to get feedback from the user, i.e. allow the user to agree or disagree with the prediction and save that feedback to make the model better in the futur.
* Add other genres to the prediction list such as folk, world music or funk
* Make the model able to guess some sub genres. Example given "Close to the Edge" by Yes, the model would return "Rock, possibly progressive rock".

## Authors
This project was done by [Michel Hijazin](https://github.com/Michell203), [Zoe Shu](https://github.com/ZoeYingShu) and [Simon Pino-Buisson](https://github.com/spynob)


