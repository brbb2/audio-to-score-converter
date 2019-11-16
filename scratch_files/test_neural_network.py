import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import os
from test_file import get_spectrogram
from sklearn.model_selection import StratifiedShuffleSplit
from music21_test import get_monophonic_ground_truth
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Flatten


def get_practice_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    return x_train, y_train, x_test, y_test


def split_data(x, y, n_splits=1, test_size=0.1):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    for train_indices, test_indices in sss.split(x, y):
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
    return x_train, y_train, x_test, y_test


def show_image(image):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()


def define_model_test2():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


def train_model(name, x_train, y_train, epochs=3, saving=True):

    model = define_model_test2()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs)

    # save model to JSON and save weights to HDF5
    if saving:
        model_json = model.to_json()
        with open(f'scratch-models/{name}.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(f'scratch-models/{name}.h5')


def load_model(model_name):
    json_file = open(f'scratch-models/{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f'scratch-models/{model_name}.h5')
    return loaded_model


def get_predictions(model_name, test_set):
    model = load_model(model_name)
    return model.predict([test_set])


def display_predictions(predictions, index=False):
    if index:
        return [np.argmax(predictions[x]) for x in range(len(predictions))]
    else:
        return [predictions[x][np.argmax(predictions[x])] for x in range(len(predictions))]


def evaluate_model(model_name, x_test, y_test):
    model = load_model(model_name)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(f'\nval_loss = {val_loss}\nval_acc  = {val_acc}')


def generate_randint_array(i, j):
    test_array = np.ones([i, j])
    for a in range(i):
        for b in range(j):
            test_array[a, b] = random.randint(0, 10)
    print(test_array)
    print(display_predictions(test_array))
    print(display_predictions(test_array, index=True))


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(y_train)
    # train_model('test2', x_train, y_train)
    # print(display_predictions(get_predictions('test2', x_test), index=True))
    # evaluate_model('test2', x_test, y_test)
    # show_image(x_test[0])


if __name__ == "__main__":
    main()
