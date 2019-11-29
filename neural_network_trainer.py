import os
import random
import numpy as np
from audio_processor import get_spectrogram, get_periodograms
from neural_network_definitions import *
from sklearn.model_selection import StratifiedShuffleSplit
from ground_truth_converter import get_monophonic_ground_truth
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import normalize, to_categorical
from keras.models import model_from_json
from keras.callbacks import TensorBoard


def print_normalisations(x):
    print('x:')
    print(x)
    print('\nnormalize(x, axis=0):')
    print(normalize(x, axis=0))
    print('\nnormalize(x, axis=1):')
    print(normalize(x, axis=1))


def print_shapes(x, y, x_train, y_train, x_test, y_test):
    print(f'      x.shape: {str(x.shape): >12}          y.shape: {str(y.shape): >7}')
    print(f'x_train.shape: {str(x_train.shape): >12}    y_train.shape: {str(y_train.shape): >7}')
    print(f' x_test.shape: {str(x_test.shape): >12}     y_test.shape: {str(y_test.shape): >7}')


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


def get_data(encoding='one_hot'):
    x_list = list()
    y_list = list()

    # for each wav file in the data directory
    for filename in os.listdir('wav_files'):
        # get the spectrogram of the audio file
        spectrum, frequencies, t, _ = get_spectrogram(f'wav_files/{filename}')
        # print(f'len(frequencies): {len(frequencies)}')
        # print(spectrum.shape, len(t))
        # and get the ground-truth note for each periodogram in the spectrum
        filename, _ = os.path.splitext(filename)  # remove file extension from the filename
        ground_truth = get_monophonic_ground_truth(f'wav_files/{filename}.wav',
                                                   f'xml_files/{filename}.musicxml',
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


def split_data(x, y, n_splits=1, test_size=0.1, random_state=42, printing=False):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    # x_train, y_train, x_test, y_test = None  # declare arrays
    for train_indices, test_indices in sss.split(x, y):
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

    if printing:
        print_shapes(x, y, x_train, y_train, x_test, y_test)
        print()
        print_counts_table(y, y_train, y_test)

    return x_train, y_train, x_test, y_test


def split_development_data_files(x, y, development_proportion=0.1):
    return x, y


def get_and_split_data(encoding='one_hot', n_splits=1, test_size=0.1, random_state=42, normalising=True, printing=False):
    x, y = get_data(encoding=encoding)
    if normalising:
        x = normalize(x, axis=0)
    if printing:
        print_normalisations(x)
    return split_data(x, y, n_splits=n_splits, test_size=test_size, random_state=random_state, printing=printing)


def train_model(model, model_name, x_train, y_train, optimizer='adam',
                loss='categorical_crossentropy', metrics=['accuracy'], epochs=2, saving=True):

    tensorboard = TensorBoard(log_dir=f'logs/{model_name}')
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x_train, y_train, epochs=epochs, callbacks=[tensorboard])

    if saving:
        save_model_to_json(model, model_name)  # save model to JSON and save weights to HDF5

    return model


def save_trained_model(model, model_name):
    model_json = model.to_json()
    with open(f'models/{model_name}.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(f'models/{model_name}.h5')


def save_model_to_json(model, model_name, scratch=False):
    model_json = model.to_json()
    if not scratch:
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


def get_model_definition(model_name, x_shape, printing=False):
    model = None
    if model_name == 'baseline':
        model = get_model_baseline(x_shape, printing=printing)
    elif model_name == 'baseline3':
        model = get_model_3(x_shape, printing=printing)
    elif model_name == 'baseline4':
        model = get_model_4(x_shape, printing=printing)
    elif model_name == 'baseline5':
        model = get_model_5(x_shape, printing=printing)
    elif model_name == 'baseline6':
        model = get_model_6(x_shape, printing=printing)
    elif model_name == 'baseline7':
        model = get_model_7(x_shape, printing=printing)

    return model


def evaluate(model, x_test, y_test, printing=True):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    val_loss, val_acc = model.evaluate(x_test, y_test)
    if printing:
        print(f'\nval_loss = {val_loss}\nval_acc  = {val_acc}')
    return val_loss, val_acc


def load_saved_model_and_evaluate(model_name, x_test, y_test, printing=True):
    model = load_model(model_name)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    val_loss, val_acc = model.evaluate(x_test, y_test)
    if printing:
        print(f'\nval_loss = {val_loss}\nval_acc  = {val_acc}')
    return val_loss, val_acc


def shuffle_data(x, y):
    np.random.seed(42)
    np.random.shuffle(x)
    np.random.seed(42)
    np.random.shuffle(y)
    return x, y


def cross_validate(model, model_name, x_dev, y_dev, n_folds=10, shuffling=True):

    fold_accuracies = np.zeros(n_folds)

    for fold in range(n_folds):

        training_files, validation_files = split_development_data_files(x_dev, y_dev)

        x_train = list()
        y_train = list()

        for wav_file in training_files:
            periodograms = get_periodograms(wav_file)
            x_train.append(periodograms)
            ground_truth = get_monophonic_ground_truth(wav_file)
            y_train.append(ground_truth)

        x_val = list()
        y_val = list()

        for wav_file in validation_files:
            periodograms = get_periodograms(wav_file)
            x_val.append(periodograms)
            ground_truth = get_monophonic_ground_truth(wav_file)
            y_val.append = ground_truth

        if shuffling:
            x_train, y_train = shuffle_data(x_train, y_train)

        train_model(model, x_train, y_train)
        accuracy = evaluate(model, x_val, y_val, printing=False)
        fold_accuracies[fold] = accuracy
        save_trained_model(model, f'{model_name}_{fold}')

    return fold_accuracies


def main():
    x_train, y_train, x_test, y_test = get_and_split_data(encoding='midi_pitch')
    y_train = y_train - 20
    y_test = y_test - 20
    x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)  # ( 105, 8193,  1)
    y_train_one_hot = to_categorical(y_train, num_classes=89, dtype='float32')

    print(f'         x_train.shape: {x_train.shape}')
    print(f'x_train_reshaped.shape: {x_train_reshaped.shape}')
    print(f'         y_train.shape: {y_train.shape}')
    print(f' y_train_one_hot.shape: {y_train_one_hot.shape}\n')

    model = get_model_definition('baseline7', x_train_reshaped.shape, printing=True)
    train_model(model, 'baseline7', x_train_reshaped, y_train_one_hot, epochs=5)


if __name__ == "__main__":
    main()
