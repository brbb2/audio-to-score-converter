from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten


def get_model_baseline(x_train_shape, printing=False):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=4, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=32))

    model.add(Conv1D(filters=8, kernel_size=64))
    model.add(MaxPool1D(pool_size=16))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')

    return model
