import os
import random
from test_functions import *
from audio_processor import get_spectrogram
from neural_network_definitions import get_model_baseline
from sklearn.model_selection import StratifiedShuffleSplit
from ground_truth_converter import get_monophonic_ground_truth
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_data(encoding='one_hot'):
    x_list = list()
    y_list = list()

    # for each wav file in the data directory
    for filename in os.listdir('scratch-wav-files'):
        # get the spectrogram of the audio file
        spectrum, frequencies, t, _ = get_spectrogram(f'scratch-wav-files/{filename}')
        # print(f'len(frequencies): {len(frequencies)}')
        # print(spectrum.shape, len(t))
        # and get the ground-truth note for each periodogram in the spectrum
        ground_truth = get_monophonic_ground_truth(f'scratch-wav-files/{filename}',
                                                   f'scratch-xml-files/{filename[:-4]}.musicxml',
                                                   encoding=encoding)
        # add each periodogram and its corresponding note to x_list and y_list respectively
        for i in range(len(ground_truth)):
            x_list.insert(0, spectrum[:, i])
            y_list.insert(0, ground_truth[i])

    # shuffle the lists, preserving the correspondence between the indices of both lists
    helper_list = list(zip(x_list, y_list))
    random.shuffle(helper_list)
    x_list, y_list = zip(*helper_list)

    # turn the lists into arrays
    x = np.array(x_list)
    y = np.array(y_list)
    return x, y


def split_data(x, y, n_splits=1, test_size=0.1, printing=False):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    x_train, y_train, x_test, y_test = np.array()  # declare arrays
    for train_indices, test_indices in sss.split(x, y):
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

    if printing:
        print_shapes(x, y, x_train, y_train, x_test, y_test)
        print()
        print_counts_table(y, y_train, y_test)

    return x_train, y_train, x_test, y_test


def get_and_split_data(encoding='one_hot', n_splits=1, test_size=0.1, normalising=True, printing=False):
    x, y = get_data(encoding=encoding)
    if normalising:
        x = normalize(x, axis=0)
    if printing:
        print_normalisations(x)
    return split_data(x, y, n_splits=n_splits, test_size=test_size, printing=printing)


def train_model(model, model_name, x_train, y_train, optimizer='adam',
                loss='sparse_categorical_crossentropy', metrics=['accuracy'], epochs=3, saving=True):

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x_train, y_train, epochs=epochs)

    if saving:
        save_model_to_json(model, model_name)  # save model to JSON and save weights to HDF5

    return model


def save_model_to_json(model, model_name, scratch=False):
    model_json = model.to_json()
    if scratch:
        with open(f'models/{model_name}.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(f'models/{model_name}.h5')
    else:
        with open(f'scratch-models/{model_name}.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(f'scratch-models/{model_name}.h5')


def load_model(model_name):
    json_file = open(f'models/{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f'models/{model_name}.h5')
    return loaded_model


def get_predictions(model, test_set):
    return model.predict([test_set])


def load_model_and_get_predictions(model_name, test_set):
    model = load_model(model_name)
    return model.predict([test_set])


def evaluate_model(model_name, x_test, y_test, printing=True):
    model = load_model(model_name)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    val_loss, val_acc = model.evaluate(x_test, y_test)
    if printing:
        print(f'\nval_loss = {val_loss}\nval_acc  = {val_acc}')
    return val_loss, val_acc


def main():
    x_train, y_train, x_test, y_test = get_and_split_data()
    # print(x_train)
    # print(y_train)
    x_train_shape = x_train.shape  # ( 105, 8193) 105 samples, each with 8193 frequency bins
    y_train_shape = y_train.shape  # ( 105,   89) 105 samples, each with 89 one-hot values
    x_train_reshaped = x_train.reshape(x_train_shape[0], x_train_shape[1], 1)  # ( 105, 8193,  1)
    y_train_reshaped = y_train.reshape(y_train_shape[0], y_train_shape[1], 1)  # ( 105,   89,  1)
    print(f'         x_train.shape: {x_train.shape}')
    print(f'x_train_reshaped.shape: {x_train_reshaped.shape}')
    print(f'         y_train.shape: {y_train.shape}')
    print(f'y_train_reshaped.shape: {y_train_reshaped.shape}')
    model = get_model_baseline(x_train_reshaped.shape)
    train_model(model, 'baseline', x_train_reshaped, y_train_reshaped)


if __name__ == "__main__":
    main()
