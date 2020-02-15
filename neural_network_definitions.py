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


def get_model_3(x_train_shape, printing=False):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=64, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=16))

    model.add(Conv1D(filters=8, kernel_size=128))
    model.add(MaxPool1D(pool_size=16))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_4(x_train_shape, printing=False):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=64, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=16))

    model.add(Conv1D(filters=8, kernel_size=256))
    model.add(MaxPool1D(pool_size=16))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_5(x_train_shape, printing=False):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=2048, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=16))

    model.add(Conv1D(filters=8, kernel_size=64))
    model.add(MaxPool1D(pool_size=16))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_6(x_train_shape, printing=False):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=16, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=4))

    model.add(Conv1D(filters=8, kernel_size=32))
    model.add(MaxPool1D(pool_size=8))

    model.add(Conv1D(filters=8, kernel_size=64))
    model.add(MaxPool1D(pool_size=8))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_7(x_train_shape, printing=False):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=16, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=2))

    model.add(Conv1D(filters=8, kernel_size=32))
    model.add(MaxPool1D(pool_size=4))

    model.add(Conv1D(filters=8, kernel_size=64))
    model.add(MaxPool1D(pool_size=4))

    model.add(Conv1D(filters=8, kernel_size=128))
    model.add(MaxPool1D(pool_size=8))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_new(x_train_shape, printing=False):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=4, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=4))

    model.add(Conv1D(filters=8, kernel_size=32))
    model.add(MaxPool1D(pool_size=4))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_midi(x_train_shape, printing=False):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=4, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=2))

    model.add(Conv1D(filters=8, kernel_size=8))
    model.add(MaxPool1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def main():
    # get_model_7(x_train_shape=(None, 8193, 1), printing=True)
    get_model_midi(x_train_shape=(None, 88, 1), printing=True)


if __name__ == '__main__':
    main()
