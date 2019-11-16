from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D


def get_model_baseline():

    model = Sequential()

    model.add(Conv1D(filters=1024, kernel_size=4))
    model.add(MaxPool1D(pool_size=2))

    model.add(Conv1D(filters=512, kernel_size=16))
    model.add(MaxPool1D(pool_size=2))

    model.add(Conv1D(filters=178, kernel_size=64))  # (2 * 89) output filters
    model.add(MaxPool1D(pool_size=2))

    model.add(Dense(89, activation='softmax'))

    return model
