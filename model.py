import numpy
from sklearn.model_selection import train_test_split

from data_reader import *
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense


class Sc2Network():

    def __init__(self, file=""):
        if file != "":
            self.model = load_model(file)
        else:
            self.model = self._create_model()

    @staticmethod
    def _create_model():
        model = Sequential()
        model.add(Dense(units=128, activation='relu', input_dim=(107)))
        model.add(Dense(128))
        model.add(Dense(128))
        model.add(Dense(units=7, activation='softmax'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train_model(self, dataFile='replays/replays_formated_data.txt', epochs=5, batch_size=32, min_score=25, verbose=0):
        X, y = get_training_data_from_file(dataFile, min_score)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        score = self.model.evaluate(X_test, y_test, batch_size=128)
        print("Model's score on test set:", score)

    def save_model(self, file):
        self.model.save(file)

    def predict(self, x):
        return self.model.predict(x)


#nn = Sc2Network()
#nn.train_model(epochs=5, batch_size=128,verbose=1)
#nn.save_model('model')