import numpy as np
from sklearn.model_selection import train_test_split

from data_reader import *

from tensorflow import keras
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization


class Sc2Network():

    def __init__(self, file=""):
        self.inputSize = (155,)
        if file != "":
            self.model = load_model(file + ".h5")
            self.model.summary()
            self.aModel2 = load_model(file + "2" + ".h5")
            self.aModel3 = load_model(file + "3" + ".h5")
            self.aModel4 = load_model(file + "4" + ".h5")
            self.aModel12 = load_model(file + "12" + ".h5")
            self.aModel331 = load_model(file + "331" + ".h5")

        else:
            self.model = self._create_model()
            self._create_arg_model_2()
            self._create_arg_model_3()
            self._create_arg_model_4()
            self._create_arg_model_12()
            self._create_arg_model_331()

    # main
    def _create_model(self):
        inp = Input(self.inputSize)
        BNorm = BatchNormalization()(inp)
        d0 = Dense(1024, activation='relu')(BNorm)
        drop0 = Dropout(0.2)(d0)
        d1 = Dense(512, activation='tanh')(drop0)
        drop1 = Dropout(0.25)(d1)
        d2 = Dense(256, activation='tanh')(drop1)
        drop2 = Dropout(0.1)(d2)
        d3 = Dense(128, activation='tanh')(drop2)
        drop3 = Dropout(0.1)(d3)
        d4 = Dense(64, activation='tanh')(drop3)
        lOut = Dense(6, activation='softmax', name="actionLayer")(d4)
        model = Model(inputs=inp, outputs=lOut)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model

    # select point
    def _create_arg_model_2(self):
        inp = Input(self.inputSize)
        BNorm = BatchNormalization()(inp)
        d0 = Dense(256, activation='relu')(BNorm)
        d1 = Dense(128, activation='tanh')(d0)
        drop1 = Dropout(0.1)(d1)
        d2 = Dense(64, activation='tanh')(drop1)
        d3 = Dense(32, activation='tanh')(d2)
        lOut = Dense(3, activation='sigmoid', name="actionLayer")(d3)
        self.aModel2 = Model(inputs=inp, outputs=lOut)
        self.aModel2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # select rect
    def _create_arg_model_3(self):
        inp = Input(self.inputSize)
        BNorm = BatchNormalization()(inp)
        d0 = Dense(256, activation='relu')(BNorm)
        d1 = Dense(128, activation='tanh')(d0)
        drop1 = Dropout(0.1)(d1)
        d2 = Dense(64, activation='tanh')(drop1)
        d3 = Dense(32, activation='tanh')(d2)
        lOut = Dense(5, activation='sigmoid', name="actionLayer")(d3)
        self.aModel3 = Model(inputs=inp, outputs=lOut)
        self.aModel3.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # control group
    def _create_arg_model_4(self):
        inp = Input(self.inputSize)
        BNorm = BatchNormalization()(inp)
        d0 = Dense(256, activation='relu')(BNorm)
        d1 = Dense(128, activation='relu')(d0)
        #drop1 = Dropout(0.1)(d1)
        d2 = Dense(64, activation='relu')(d1)
        d3 = Dense(32, activation='relu')(d2)
        lOut = Dense(2, activation='sigmoid')(d3)
        self.aModel4 = Model(inputs=inp, outputs=lOut)
        self.aModel4.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # attack
    def _create_arg_model_12(self):
        inp = Input(self.inputSize)
        BNorm = BatchNormalization()(inp)
        d0 = Dense(256, activation='relu')(BNorm)
        d1 = Dense(128, activation='relu')(d0)
        drop1 = Dropout(0.1)(d1)
        d2 = Dense(64, activation='relu')(drop1)
        d3 = Dense(32, activation='relu')(d2)
        lOut = Dense(3, activation='sigmoid')(d3)
        self.aModel12 = Model(inputs=inp, outputs=lOut)
        self.aModel12.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # move
    def _create_arg_model_331(self):
        inp = Input(self.inputSize)
        BNorm = BatchNormalization()(inp)
        d0 = Dense(512, activation='relu')(BNorm)
        d1 = Dense(256, activation='relu')(d0)
        drop1 = Dropout(0.25)(d1)
        d2 = Dense(128, activation='tanh')(drop1)
        drop2 = Dropout(0.1)(d2)
        d3 = Dense(64, activation='tanh')(drop2)
        lOut = Dense(3, activation='sigmoid')(d3)
        self.aModel331 = Model(inputs=inp, outputs=lOut)
        self.aModel331.compile(optimizer='adam', loss='mean_squared_error',
                               metrics=['accuracy'])

    def train_model(self, epochs=5, batch_size=32, min_score=25,
                    verbose=0):
        X, y, X2, y2 = get_training_data_from_file(min_score, 35)
        # Training the main model first
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        score1 = self.model.evaluate(X_test, y_test, batch_size=batch_size)

        # Training the argument models
        scores = []
        for i, m in enumerate(
                [self.aModel2, self.aModel3, self.aModel4, self.aModel331, None, self.aModel12]):
            if m is None:
                continue
            X_train, X_test, y_train, y_test = train_test_split(np.asarray(X2[i]), np.asarray(y2[i]))
            m.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
            scores.append(m.evaluate(X_test, y_test, batch_size=batch_size))

        print("Main Model's score on test set:", score1)
        print("Other models:")
        for i in scores:
            print(i)

    def save_model(self, file):
        self.model.save(file + ".h5")
        self.aModel2.save(file + "2" + ".h5")
        self.aModel3.save(file + "3" + ".h5")
        self.aModel4.save(file + "4" + ".h5")
        self.aModel12.save(file + "12" + ".h5")
        self.aModel331.save(file + "331" + ".h5")

    def predict(self, x):
        m = np.argmax(self.model.predict(x))
        fid = 0
        a = []
        if m == 0:
            fid = 2
            a = self.aModel2.predict(x)[0]
            a = [[int(a[0] * 9)], [a[1] * 79, a[2] * 64]]
        if m == 1:
            fid = 3
            a = self.aModel3.predict(x)[0]
            a = [[int(a[0] * 9)], [a[1] * 79, a[2] * 64], [a[3] * 79, a[4] * 64]]
        if m == 2:
            fid = 4
            a = self.aModel4.predict(x)[0]
            a = [[int(a[0] * 9)], [int(a[1] * 9)]]
        if m == 3:
            fid = 331
            a = self.aModel331.predict(x)[0]
            a = [[int(a[0] * 9)], [a[1] * 79, a[2] * 64]]
        if m == 4:
            fid = 333
            a = self.aModel331.predict(x)[0]
            a = [[int(a[0] * 9)], [a[1] * 79, a[2] * 64]]
        if m == 5:
            fid = 12
            a = self.aModel12.predict(x)[0]
            a = [[int(a[0] * 9)], [a[1] * 79, a[2] * 64]]
        #print(fid, a)
        return fid, a


if __name__ == "__main__":
    nn = Sc2Network()
    nn.train_model(epochs=10, batch_size=64, verbose=1, min_score=70)
    nn.save_model('model')
