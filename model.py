import numpy
from sklearn.model_selection import train_test_split

from data_reader import *

from tensorflow import keras
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization


class Sc2Network():

    def __init__(self, file=""):
        if file != "":
            self.model = load_model(file)
            self.model.summary()
        else:
            self.model = self._create_model()

    @staticmethod
    def _create_model():
        inp = Input((171,))
        BNorm = BatchNormalization()(inp)
        d1 = Dense(256, activation='relu')(BNorm)
        drop1 = Dropout(0.1)(d1)
        d2 = Dense(256, activation='relu')(drop1)

        ld1 = Dense(128, activation='sigmoid')(d2)
        md1 = Dense(128, activation='sigmoid')(d2)
        rd1 = Dense(128, activation='sigmoid')(d2)

        ld2 = Dense(64, activation='sigmoid')(ld1)
        md2 = Dense(64, activation='sigmoid')(md1)
        rd2 = Dense(64, activation='sigmoid')(rd1)

        lOut = Dense(6, activation='softmax', name="actionLayer")(ld2)
        mOut = Dense(4, activation='sigmoid', name="CoordLayer")(md2)
        rOut = Dense(2, activation='sigmoid', name="ArgLayer")(rd2)

        model = Model(inputs=[inp], outputs=[lOut, mOut, rOut])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', 'accuracy', 'accuracy'])
        return model

    def train_model(self, dataFile='replays/data.txt', epochs=5, batch_size=32, min_score=25,
                    verbose=0):
        X, y1, y2, y3 = get_training_data_from_file(dataFile, min_score)
        X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(X, y1, y2, y3)
        y_train = [np.asarray(y1_train), np.asarray(y2_train), np.asarray(y3_train)]
        y_test = [np.asarray(y1_test), np.asarray(y2_test), np.asarray(y3_test)]
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        score = self.model.evaluate(X_test, y_test, batch_size=batch_size)
        print("Model's score on test set:", score)

    def save_model(self, file):
        self.model.save(file)

    def predict(self, x):
        return self.model.predict(x)

if __name__ == "__main__":
    nn = Sc2Network()
    nn.train_model(epochs=5, batch_size=128, verbose=1, min_score=0)
    nn.save_model('model.h5')
