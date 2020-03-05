from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, LSTM, Dropout


def print_model_summary(model, default=True):
    if default:
        model.summary()
    else:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')


def get_model_baseline(x_train_shape, printing=False, default=True):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=4, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=32))

    model.add(Conv1D(filters=8, kernel_size=64))
    model.add(MaxPool1D(pool_size=16))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        print_model_summary(model, default)

    return model


def get_model_3(x_train_shape, printing=False, default=True):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=64, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=16))

    model.add(Conv1D(filters=8, kernel_size=128))
    model.add(MaxPool1D(pool_size=16))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        print_model_summary(model, default)

    return model


def get_model_4(x_train_shape, printing=False, default=True):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=64, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=16))

    model.add(Conv1D(filters=8, kernel_size=256))
    model.add(MaxPool1D(pool_size=16))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        print_model_summary(model, default)

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


def get_model_freq_dense(x_train_shape, printing=False):

    model = Sequential()

    model.add(Flatten(input_shape=x_train_shape[1:]))
    model.add(Dense(512))

    model.add(Dense(256))

    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_midi_dense(x_train_shape, printing=False):

    model = Sequential()

    model.add(Flatten(input_shape=x_train_shape[1:]))
    model.add(Dense(89))

    model.add(Dense(200))

    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_new(x_train_shape, printing=False):

    model = Sequential()

    # model.add(Conv1D(filters=8, kernel_size=4, input_shape=x_train_shape[1:]))
    model.add(Conv1D(filters=128, kernel_size=64, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=8))

    model.add(Conv1D(filters=128, kernel_size=32))
    # model.add(Conv1D(filters=2, kernel_size=32))
    model.add(MaxPool1D(pool_size=4))

    model.add(Conv1D(filters=128, kernel_size=8, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=4))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_new_dropout(x_train_shape, dropout_rate=0.2, printing=False):

    model = Sequential()

    model.add(Dropout(dropout_rate, input_shape=x_train_shape[1:]))
    model.add(Conv1D(filters=8, kernel_size=4))
    model.add(MaxPool1D(pool_size=4))

    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters=8, kernel_size=32))
    model.add(MaxPool1D(pool_size=4))

    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_midi(x_train_shape, printing=False):

    model = Sequential()

    # model.add(Conv1D(filters=8, kernel_size=4, input_shape=x_train_shape[1:]))
    model.add(Conv1D(filters=128, kernel_size=32, input_shape=x_train_shape[1:]))
    model.add(MaxPool1D(pool_size=2))

    # model.add(Conv1D(filters=8, kernel_size=8))
    model.add(Conv1D(filters=128, kernel_size=8))
    model.add(MaxPool1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax'))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_midi_dropout(x_train_shape, dropout_rate=0, printing=False):

    model = Sequential()

    model.add(Conv1D(filters=8, kernel_size=4, input_shape=x_train_shape[1:], dropout_rate=dropout_rate))
    model.add(MaxPool1D(pool_size=2))

    model.add(Conv1D(filters=8, kernel_size=8))
    model.add(MaxPool1D(pool_size=2))

    # model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(89, activation='softmax', dropout_rate=dropout_rate))

    if printing:
        for layer in model.layers:
            print(f'{layer.__class__.__name__}:\n{layer.input_shape} -> {layer.output_shape}\n')
        print()

    return model


def get_model_midi_rnn(x_train_shape, printing=False, default=True):

    model = Sequential()

    model.add(LSTM(128, input_shape=x_train_shape[1:]))

    model.add(Dense(90, activation='softmax'))  # 1 rest category, 88 MIDI-pitch categories and 1 End-of-File category

    if printing:
        print_model_summary(model, default)

    return model


def main():
    # get_model_7(x_train_shape=(None, 8193, 1), printing=True)
    # model = get_model_midi(x_train_shape=(None, 88, 1), printing=True)
    model = get_model_midi_rnn(x_train_shape=(None, 88, 1), printing=True, default=False)
    model.summary()


if __name__ == '__main__':
    main()
