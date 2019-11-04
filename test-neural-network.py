# import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Flatten


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    return x_train, y_train, x_test, y_test


def show_image(image):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()


def train_and_save_model(name, x_train, y_train):
    # x_train, y_train, x_test, y_test = get_data()
    _, _, x_test, y_test = get_data()

    # define model
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit model
    model.fit(x_train, y_train, epochs=3)

    # save model
    # model.save(f'scratch-models/{name}.model')

    # serialize model to JSON
    model_json = model.to_json()
    with open(f'scratch-models/{name}.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f'scratch-models/{name}.h5')
    '''
    # evaluate model
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(f'\nval_loss = {val_loss}\nval_acc = {val_acc}')

    # use model to make predictions
    predictions = model.predict([x_test])
    print(np.argmax(predictions[0]))
    '''


def evaluate_model(model_name, x_test, y_test):
    # model = tf.keras.models.load_model(f'scratch-models/{model_name}.model')
    # val_loss, val_acc = model.evaluate(x_test, y_test)
    # print(f'val_loss = {val_loss}\nval_acc{val_acc}')
    json_file = open(f'scratch-models/{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f'scratch-models/{model_name}.h5')
    predictions = loaded_model.predict([x_test])
    print(np.argmax(predictions[0]))


def main():
    x_train, y_train, x_test, y_test = get_data()
    # train_and_save_model('test2', x_train, y_train)
    evaluate_model('test2', x_test, y_test)
    # show_image(x_test[1])


if __name__ == "__main__":
    main()
