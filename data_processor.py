import numpy as np
from math import ceil
from os import listdir
from os.path import isfile, splitext
from sklearn.model_selection import StratifiedShuffleSplit
from audio_processor import get_window_parameters, get_spectrogram
from ground_truth_converter import get_monophonic_ground_truth
from midi_manager import encode_ground_truth_array
from neural_network_trainer import print_shapes, print_counts_table, make_dictionary_from_arrays, flatten_split_data,\
    reshape, split_dictionary, save_data_arrays


def load_audio_files_and_get_ground_truth(window_size, splitting_on_file_name=False, using_midi_bins=False,
                                          fft_strategy='scipy'):

    x_list = list()
    y_list = list()
    sources = list()

    nperseg, noverlap = get_window_parameters(window_size)

    # for each wav file in the data directory
    for file_name in listdir('wav_files'):
        if isfile(f'wav_files/{file_name}'):

            # get the spectrogram of the audio file
            _, _, spectrogram = get_spectrogram(f'wav_files/{file_name}', using_midi_bins=using_midi_bins,
                                                nperseg=nperseg, noverlap=noverlap, strategy=fft_strategy)

            # and get the ground-truth note for each periodogram in the spectrum
            file_name, _ = splitext(file_name)  # remove file extension from the filename
            ground_truth = get_monophonic_ground_truth(f'wav_files/{file_name}.wav', f'xml_files/{file_name}.musicxml',
                                                       encoding=None,
                                                       nperseg=nperseg, noverlap=noverlap)

            # add each periodogram and its corresponding note to x_list and y_list respectively,
            # inserting data at the front of the lists for efficiency
            for i in range(len(ground_truth)):
                x_list.insert(0, spectrogram[:, i])
                y_list.insert(0, ground_truth[i])
                if splitting_on_file_name:
                    sources.insert(0, file_name)

    # turn the lists into arrays, reversing them to reverse the effects of inserting at the front of the lists
    x = np.array(x_list)[::-1]
    y = np.array(y_list)[::-1]
    if splitting_on_file_name:
        sources = np.array(sources)[::-1]

    if splitting_on_file_name:
        return x, y, sources
    else:
        return x, y


def add_spectral_powers(x, printing=False):
    x_squared = np.apply_along_axis(np.square, 1, x)
    spectral_powers = np.apply_along_axis(np.sum, 1, x_squared)
    spectral_powers = spectral_powers.reshape(spectral_powers.shape[0], 1)
    x_with_powers = np.concatenate((x, spectral_powers), axis=1)
    if printing:
        print(f'              x.shape: {x.shape}')
        print(f'spectral_powers.shape: {spectral_powers.shape}')
        print(f'  x_with_powers.shape: {x_with_powers.shape}\n\n')
    return x_with_powers


def add_first_order_differences(x, y, sources, printing=False):
    # this function is indifferent to whether or not x has already been reshaped to (x.shape[0], x.shape[1], 1)

    # initialise lists for accumulating data
    x_two_channels = list()
    y_without_firsts = list()
    sources_list = list()

    channels_already_present = len(x.shape) == 3
    periodogram_length = x.shape[1]

    i = 0
    while i < len(x):
        source = sources[i]  # record the current source file name

        i += 1  # skip the first periodogram of each file
        while i < len(x) and sources[i] == source:

            two_channel_periodogram = np.zeros(shape=(periodogram_length, 2))

            if channels_already_present:
                two_channel_periodogram[:, 0] = x[i, :, 0]
                two_channel_periodogram[:, 1] = x[i, :, 0] = x[i - 1, :, 0]
            else:
                two_channel_periodogram[:, 0] = x[i]
                two_channel_periodogram[:, 1] = x[i] - x[i - 1]

            x_two_channels.insert(0, two_channel_periodogram)
            y_without_firsts.insert(0, y[i])
            sources_list.insert(0, source)

            i += 1

    x_two_channels = np.array(x_two_channels)[::-1]
    y_without_firsts = np.array(y_without_firsts)[::-1]
    sources = np.array(sources_list)[::-1]

    if printing:
        print(f'x before: {x.shape}\n{x}\n')
        print(f'x after: {x_two_channels.shape}\n{x_two_channels}\n\n')

    return x_two_channels, y_without_firsts, sources


def balance_rests(x, y, sources=None, encoding=None, printing=False):

    # sanity-test inputs
    if sources is not None:
        assert len(sources) == len(x) == len(y)
    else:
        assert len(x) == len(y)

    y_targets, y_counts = np.unique(y, return_counts=True)
    average_non_rest_count = np.average(y_counts[1:])
    number_of_rests_to_keep = ceil(average_non_rest_count)

    rest_representation = encode_ground_truth_array(np.array(['rest']),
                                                    current_encoding=None, desired_encoding=encoding)[0]

    rest_indices = np.where(y == rest_representation)[0]
    np.random.seed(42)
    np.random.shuffle(rest_indices)
    rests_to_keep_indices = rest_indices[:number_of_rests_to_keep]
    rests_to_drop_indices = rest_indices[number_of_rests_to_keep:]

    x_new = np.delete(x, rests_to_drop_indices, axis=0)
    y_new = np.delete(y, rests_to_drop_indices, axis=0)
    if sources is not None:
        sources = np.delete(sources, rests_to_drop_indices, axis=0)

    if printing:
        print(f'        number of rests: {rest_indices.size}')
        print(f'number of rests to keep: {number_of_rests_to_keep}')
        print(f'                  split: {rests_to_keep_indices.size} | {rests_to_drop_indices.size}\n')
        print(f'    x: {x.shape}\n{x}\n')
        print(f'x_new: {x_new.shape}\n{x_new}\n')
        print(f'    y: {y.shape}\n{y}\n')
        print(f'y_new: {y_new.shape}\n{y_new}\n')

    if sources is None:
        return x_new, y_new
    else:
        return x_new, y_new, sources


def normalise(x):
    return x


def encode(y, current_encoding=None, target_encoding='label'):
    y = encode_ground_truth_array(y, current_encoding=current_encoding, desired_encoding=target_encoding)
    return y


def split(x, y, n_splits=1, test_size=0.1, printing=False):

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    x_train = None
    y_train = None
    x_val = None
    y_val = None

    for train_indices, val_indices in sss.split(x, y):
        x_train, x_val = x[train_indices], x[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

    if printing:
        print()
        print_shapes(x, y, x_train, y_train, x_val, y_val)
        print()
        print_counts_table(y, y_train, y_val)

    return x_train, y_train, x_val, y_val  # flat


def split_on_file_names(dictionary):
    x_train, y_train, x_val, y_val = split_dictionary(dictionary)
    return x_train, y_train, x_val, y_val  # not flat


def get_data(window_size, splitting_on_file_name=True, adding_spectral_powers=True, adding_first_order_differences=True,
             balancing_rests=True, normalising=True, target_encoding='label', adding_file_separators=True,
             saving=False, save_name=None, printing=False):

    sources = None

    if splitting_on_file_name or adding_first_order_differences:
        x, y, sources = load_audio_files_and_get_ground_truth(window_size,
                                                              splitting_on_file_name=splitting_on_file_name)
    else:
        x, y = load_audio_files_and_get_ground_truth(window_size, splitting_on_file_name=splitting_on_file_name)

    if adding_spectral_powers:
        x = add_spectral_powers(x, printing=printing)

    if adding_first_order_differences:
        x, y, sources = add_first_order_differences(x, y, sources, printing=printing)
    else:
        x = reshape(x)  # set up x for a single channel if it has not been so already

    if balancing_rests:
        if splitting_on_file_name:
            x, y, sources = balance_rests(x, y, sources)
        else:
            x, y = balance_rests(x, y)

    if normalising:  # must be written to work for channels
        x = normalise(x)

    y = encode(y, target_encoding)

    if splitting_on_file_name:
        dictionary = make_dictionary_from_arrays(x, y, sources)
        x_train, y_train, x_val, y_val = split_on_file_names(dictionary)
        x_train, y_train, x_val, y_val = flatten_split_data(x_train, y_train, x_val, y_val, adding_file_separators)
    else:
        x_train, y_train, x_val, y_val = split(x, y)

    if saving and save_name is not None:
        save_data_arrays(x_train, y_train, x_val, y_val, version=save_name)

    return x_train, y_train, x_val, y_val


def main():
    get_data(50, printing=True)


if __name__ == '__main__':
    main()
