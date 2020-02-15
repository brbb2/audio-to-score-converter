import os
import random
import numpy as np
import midi_manager
from math import floor
from audio_processor import get_spectrogram_scipy, get_periodograms
from neural_network_definitions import *
from sklearn.model_selection import StratifiedShuffleSplit
from ground_truth_converter import get_monophonic_ground_truth
from keras.utils import normalize, to_categorical
from keras.models import model_from_json
from keras.callbacks import TensorBoard, EarlyStopping


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


def print_normalisation_options(x):
    print('x:')
    print(x)
    print('\nkeras.utils.normalize(x, axis=0):')
    print(normalize(x, axis=0))
    print('\nkeras.utils.normalize(x, axis=1):')
    print(normalize(x, axis=1))


def normalise_x(x, axis=0, printing=False, printing_normalisation_options=False):
    x_normalised = normalize(x, axis=axis)
    if printing:
        print()
        print(x_normalised)
    if printing_normalisation_options:
        print_normalisation_options(x)
    return x_normalised


def normalise_x_train_and_x_val(x_train, x_val, axis=0, printing=False):
    separation_index = x_train.shape[axis]
    x = np.concatenate((x_train, x_val), axis=axis)
    x_normalised = normalize(x, axis=axis)
    x_train_normalised = x_normalised[:separation_index]
    x_val_normalised = x_normalised[separation_index:]
    if printing:
        print()
        print(f'           x_train.shape: {x_train.shape}    '
              f'             x_val.shape: {x_val.shape}\n'
              f'x_train_normalised.shape: {x_train_normalised.shape}    '
              f'x_train_normalised.shape: {x_val_normalised.shape}')
        print()
        print(x_normalised)
    return x_train_normalised, x_val_normalised


def get_data(encoding='midi_pitch', midi_bins=False, nperseg=4096, noverlap=2048):
    x_list = list()
    y_list = list()

    # for each wav file in the data directory
    for filename in os.listdir('wav_files'):
        # get the spectrogram of the audio file
        f, t, sxx = get_spectrogram_scipy(f'wav_files/{filename}', midi_bins=midi_bins,
                                          nperseg=nperseg, noverlap=noverlap)
        # and get the ground-truth note for each periodogram in the spectrum
        filename, _ = os.path.splitext(filename)  # remove file extension from the filename
        ground_truth = get_monophonic_ground_truth(f'wav_files/{filename}.wav',
                                                   f'xml_files/{filename}.musicxml',
                                                   encoding=encoding, nperseg=nperseg, noverlap=noverlap)
        # add each periodogram and its corresponding note to x_list and y_list respectively
        for i in range(len(ground_truth)):
            x_list.insert(0, sxx[:, i])
            y_list.insert(0, ground_truth[i])

    # shuffle the lists, preserving the correspondence between the indices of both lists
    helper_list = list(zip(x_list, y_list))
    random.shuffle(helper_list)
    x_list, y_list = zip(*helper_list)

    # turn the lists into arrays
    x = np.array(x_list)
    y = np.array(y_list)
    return x, y


def reduce_data(x, y, proportion=0.25):
    assert (0 <= proportion <= 1)
    size = floor(x.shape[0] * proportion)
    return x[:size], y[:size]


def shuffle_data(x, y):
    np.random.seed(42)
    np.random.shuffle(x)
    np.random.seed(42)
    np.random.shuffle(y)
    return x, y


def split_data(x, y, n_splits=1, test_size=0.1, random_state=42, printing=False):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    for train_indices, test_indices in sss.split(x, y):
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

    if printing:
        print()
        print_shapes(x, y, x_train, y_train, x_test, y_test)
        print()
        print_counts_table(y, y_train, y_test)

    return x_train, y_train, x_test, y_test


def preprocess_data(x_train, y_train, x_val, y_val, encoding='midi_pitch',
                    normalising_features=True, axis=0, start=21, printing=False):

    if normalising_features:
        x_train, x_val = normalise_x_train_and_x_val(x_train, x_val, axis=axis)

    if printing:
        x_train_original = np.array(x_train)
        y_train_original = np.array(y_train)
        x_val_original = np.array(x_val)
        y_val_original = np.array(y_val)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

    y_train = y_train - start + 1
    mask_train = np.where(y_train == midi_manager.REST_ENCODING - start + 1)
    y_train[mask_train] = 0

    y_val = y_val - start + 1
    mask_val = np.where(y_val == midi_manager.REST_ENCODING - start + 1)
    y_val[mask_val] = 0

    if encoding == 'one_hot':
        y_train = to_categorical(y_train, num_classes=89, dtype='float32')
        y_val = to_categorical(y_val, num_classes=89, dtype='float32')

    if printing:
        print(x_train_original)
        print(x_train)
        print(y_train_original)
        print(y_train)
        print()
        print(f'         x_train.shape: {x_train_original.shape}')
        print(f'x_train_reshaped.shape: {x_train.shape}')
        print(f'         y_train.shape: {y_train_original.shape}')
        print(f' y_train_one_hot.shape: {y_train.shape}\n')
        print()
        print(x_val_original)
        print(x_val)
        print(y_val_original)
        print(y_val)
        print()
        print(f'           x_val.shape: {x_val_original.shape}')
        print(f'  x_val_reshaped.shape: {x_val.shape}')
        print(f'           y_val.shape: {y_val_original.shape}')
        print(f'   y_val_one_hot.shape: {y_val.shape}\n')

    return x_train, y_train, x_val, y_val


def get_and_prepare_data(encoding='midi_pitch', n_splits=1, test_size=0.1, random_state=42,
                         nperseg=4096, noverlap=2048, midi_bins=False, normalising_features=True,
                         printing=False, saving=False, version=None):
    x, y = get_data(encoding=encoding, midi_bins=midi_bins, nperseg=nperseg, noverlap=noverlap)
    x_train, y_train, x_val, y_val = split_data(x, y, n_splits=n_splits, test_size=test_size,
                                                random_state=random_state, printing=printing)
    x_train, y_train, x_val, y_val = preprocess_data(x_train, y_train, x_val, y_val,
                                                     normalising_features=normalising_features)

    if saving and version is not None:
        save_data_arrays(x_train, y_train, x_val, y_val, version)

    return x_train, y_train, x_val, y_val


def print_data(x_train, y_train, x_val, y_val):
    print(f'x_train: {x_train.shape}')
    print(x_train)
    print(f'\ny_train: {y_train.shape}')
    print(y_train)
    print(f'\nx_val: {x_val.shape}')
    print(x_val)
    print(f'\ny_val: {y_val.shape}')
    print(y_val)


def save_data_arrays(x_train, y_train, x_val, y_val, version, printing=True):

    path = f'data_arrays/{version}'

    try:
        os.mkdir(path)
    except OSError:
        print(f'Creation of the directory \"{path}\" failed.')
    else:
        if printing:
            print(f'Successfully created the directory \"{path}\".')

    np.save(f'data_arrays/{version}/x_train.npy', x_train)
    np.save(f'data_arrays/{version}/y_train.npy', y_train)
    np.save(f'data_arrays/{version}/x_val.npy', x_val)
    np.save(f'data_arrays/{version}/y_val.npy', y_val)

    if printing:
        print(f'Successfully saved data arrays in the directory \"{path}\".')


def load_data_arrays(version):
    x_train = np.load(f'data_arrays/{version}/x_train.npy')
    y_train = np.load(f'data_arrays/{version}/y_train.npy')
    x_val = np.load(f'data_arrays/{version}/x_val.npy')
    y_val = np.load(f'data_arrays/{version}/y_val.npy')
    return x_train, y_train, x_val, y_val


def get_model_definition(model_name, x_shape, printing=False):
    model = None
    if model_name == 'baseline':
        model = get_model_baseline(x_shape, printing=printing)
    elif model_name == 'new':
        model = get_model_new(x_shape, printing=printing)
    elif model_name == 'midi':
        model = get_model_midi(x_shape, printing=printing)
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


def train_model(model, model_name, x_train, y_train, x_val, y_val, optimizer='adam', epochs=50, patience=2,
                loss='categorical_crossentropy', metrics=['accuracy'], min_delta=0, saving=True):

    tensorboard = TensorBoard(log_dir=f'logs/{model_name}')
    es = EarlyStopping(patience=patience, min_delta=min_delta, restore_best_weights=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x_train, y_train, epochs=epochs, callbacks=[tensorboard, es], validation_data=(x_val, y_val))

    if saving:
        save_model_to_json(model, model_name)  # save model to JSON and save weights to HDF5

    return model


def train(model_name, save_name, x_train, y_train, x_val, y_val, optimizer='adam', epochs=50, patience=2,
          loss='categorical_crossentropy', metrics=['accuracy'], min_delta=0, saving=True, printing=True):
    model = get_model_definition(model_name, x_train.shape, printing=printing)
    train_model(model, save_name, x_train, y_train, x_val, y_val, optimizer=optimizer, epochs=epochs, patience=patience,
                loss=loss, metrics=metrics, min_delta=min_delta, saving=saving)


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


def load_model_and_get_predictions(model_name, x):
    model = load_model(model_name)
    return model.predict([x])


def show_example_prediction(model_name, i=0, version=1, x_val=None, y_val=None, printing_in_full=False):
    if x_val is None or y_val is None:
        _, _, x_val, y_val = load_data_arrays(version)
    example = (x_val[i], y_val[i])
    model_input = example[0].reshape(1, example[0].shape[0], 1)
    example_prediction = load_model_and_get_predictions(model_name, model_input)[0]
    example_ground_truth = example[1]
    i_prediction = np.argmax(example_prediction)
    i_ground_truth = np.argmax(example_ground_truth)
    midi_pitch_prediction = midi_manager.interpret_one_hot(example_prediction)
    midi_pitch_ground_truth = midi_manager.interpret_one_hot(example_ground_truth)
    note_name_prediction = midi_manager.get_note_name(midi_pitch_prediction)
    note_name_ground_truth = midi_manager.get_note_name(midi_pitch_ground_truth)

    if printing_in_full:
        print(f'\nmodel_input: {model_input.shape}')
        print(model_input)
        print(f'\nexample_prediction: {example_prediction.shape}')
        print(example_prediction)
        print(f'\nexample_ground_truth: {example_ground_truth.shape}')
        print(example_ground_truth)

    print()
    print(f'val instance {i}')
    print(f'         i_prediction: {i_prediction:>4}            i_ground_truth: {i_ground_truth:>4}')
    print(f'midi_pitch_prediction: {midi_pitch_prediction:>4}   midi_pitch_ground_truth: {midi_pitch_ground_truth:>4}')
    print(f' note_name_prediction: {note_name_prediction:>4}    note_name_ground_truth: {note_name_ground_truth:>4}')


def show_first_n_predictions(model_name, n=25, x_val=None, y_val=None, version=1, printing_in_full=False):
    if x_val is None or y_val is None:
        _, _, x_val, y_val = load_data_arrays(version)
    assert (n < len(x_val))
    for i in range(n):
        show_example_prediction(model_name, i, x_val=x_val, y_val=y_val, printing_in_full=printing_in_full)


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
    x_train, y_train, x_val, y_val = get_and_prepare_data(midi_bins=False, nperseg=2048, noverlap=1024,
                                                          saving=True, version='label_freq_025ms')
    x_train, y_train, x_val, y_val = load_data_arrays('label_freq_025ms')
    print_data(x_train, y_train, x_val, y_val)

    train('new', 'label_freq_025ms', x_train, y_train, x_val, y_val, printing=True,
          loss='sparse_categorical_crossentropy')

    # show_example_prediction('one_hot_freq_050ms', 2, x_val=x_val, y_val=y_val, printing_in_full=True)


if __name__ == "__main__":
    main()
