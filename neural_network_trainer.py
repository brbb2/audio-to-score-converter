import numpy as np
import random
import os
from audio_processor import get_spectrogram
from sklearn.model_selection import StratifiedShuffleSplit
from ground_truth_converter import get_monophonic_ground_truth
from neural_network_definitions import get_model_baseline
from keras.models import model_from_json


def get_data(encoding='one_hot'):
    x_list = list()
    y_list = list()

    # for each wav file in the data directory
    for filename in os.listdir('scratch-wav-files'):
        # get the spectrogram of the audio file
        spectrum, _, t, _ = get_spectrogram(f'scratch-wav-files/{filename}')
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
    for train_indices, test_indices in sss.split(x, y):
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

    if printing:
        print(f'      x.shape: {str(x.shape): >12}          y.shape: {str(y.shape): >7}')
        print(f'x_train.shape: {str(x_train.shape): >12}    y_train.shape: {str(y_train.shape): >7}')
        print(f' x_test.shape: {str(x_test.shape): >12}     y_test.shape: {str(y_test.shape): >7}')
        print()
        print_counts_table(y, y_train, y_test)

    return x_train, y_train, x_test, y_test


def get_and_split_data(encoding='one_hot', n_splits=1, test_size=0.1, printing=False):
    x, y = get_data(encoding=encoding)
    return split_data(x, y, n_splits=n_splits, test_size=test_size, printing=printing)


def print_counts_table(y, y_train, y_test):
    y_targets, y_counts = np.unique(y, return_counts=True)
    y_train_targets, y_train_counts = np.unique(y_train, return_counts=True)
    y_test_targets, y_test_counts = np.unique(y_test, return_counts=True)
    # print(y_targets, y_counts)
    # print(y_train_targets, y_train_counts)
    # print(y_test_targets, y_test_counts)
    print('value     |       y  y_train   y_test')
    for i in range(len(y_targets)):
        y_train_indices = np.where(y_train_targets == y_targets[i])[0]
        y_test_indices = np.where(y_test_targets == y_targets[i])[0]
        # print(f'{y_targets[i]} at index {y_train_indices}')
        if len(y_train_indices) > 0:
            y_train_index = y_train_indices[0]
            y_train_count = y_train_counts[y_train_index]
        else:
            y_train_count = 0
        if len(y_test_indices) > 0:
            y_test_index = y_test_indices[0]
            y_test_count = y_test_counts[y_test_index]
        else:
            y_test_count = 0
        print(f'{str(y_targets[i]): <9} | {y_counts[i]: >7}  {y_train_count: >7}  {y_test_count: >7}')


def train_model(model, model_name, x_train, y_train, epochs=3, saving=True):

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs)

    # save model to JSON and save weights to HDF5
    if saving:
        model_json = model.to_json()
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


def load_scratch_model(model_name):
    json_file = open(f'scratch-models/{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f'scratch-models/{model_name}.h5')
    return loaded_model


def get_predictions(model, test_set):
    return model.predict([test_set])


def load_model_and_get_predictions(model_name, test_set):
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


def test_getting_data(encoding='one_hot', printing=True, deep_printing=False):
    x, y = get_data(encoding=encoding)
    x_train, y_train, x_test, y_test = split_data(x, y, printing=deep_printing)
    if printing:
        print(x_train)
        print()
        print(y_train)
        print()
        print(x_test)
        print()
        print(y_test)
    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = get_and_split_data()
    model = get_model_baseline()
    train_model(model, 'baseline', x_train, y_train)


if __name__ == "__main__":
    main()
