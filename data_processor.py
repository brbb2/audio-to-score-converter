import numpy as np
from os import mkdir
from math import ceil
from os import listdir
from keras.utils import normalize
from os.path import isfile, splitext
from audio_processor import get_spectrogram
from sklearn.model_selection import StratifiedShuffleSplit
from ground_truth_converter import get_monophonic_ground_truth
from neural_network_trainer import print_shapes, print_counts_table, print_split_data
from encoder import encode_ground_truth_array, get_bof_artificial_periodogram, get_eof_artificial_periodogram,\
    BoF_LABEL_ENCODING, encode_file_name


def save_data_arrays(x_train, y_train, x_val, y_val, save_name, printing=True):

    path = f'data_arrays/{save_name}'

    try:
        mkdir(path)
    except OSError:
        print(f'Creation of the directory \"{path}\" failed.')
    else:
        if printing:
            print(f'Successfully created the directory \"{path}\".')

    np.save(f'data_arrays/{save_name}/x_train.npy', x_train)
    np.save(f'data_arrays/{save_name}/y_train.npy', y_train)
    np.save(f'data_arrays/{save_name}/x_val.npy', x_val)
    np.save(f'data_arrays/{save_name}/y_val.npy', y_val)

    if printing:
        print(f'Successfully saved data arrays in the directory \"{path}\".')


def flatten_array_of_arrays(array_of_arrays, inserting_file_separators=False, encoding=None, features=True,
                            printing=False, deep_printing=False):
    flattened_data = list()
    channels_present = len(array_of_arrays[0].shape) == 3
    number_of_channels = None
    if channels_present:
        number_of_channels = array_of_arrays[0].shape[2]
    for i in range(len(array_of_arrays)):
        for periodogram in array_of_arrays[i]:
            flattened_data.insert(0, periodogram)
        if inserting_file_separators:
            if features:
                if deep_printing:
                    print(len(array_of_arrays[0][1]))
                if channels_present:
                    flattened_data.insert(0, np.full((len(array_of_arrays[0][1]), number_of_channels), -1))
                else:
                    flattened_data.insert(0, np.full(len(array_of_arrays[0][1]), -1))
            else:
                if deep_printing:
                    print('EoF')
                end_of_file_marker = encode_ground_truth_array('EoF', current_encoding=None, desired_encoding=encoding)
                flattened_data.insert(0, end_of_file_marker)

    if printing:
        print(f'array_of_arrays[0].shape: {array_of_arrays[0].shape}')
        print(f'     len(flattened_data): {len(flattened_data)}')
        if type(flattened_data[0]) is not str:
            print(f'       flattened_data[0]: {flattened_data[0].shape}\n{flattened_data[0]}\n')
        else:
            print(f' type(flattened_data[0]): {type(flattened_data[0])}')
            print(f'       flattened_data[0]: {flattened_data[0]}\n')

    flattened_data = np.array(flattened_data)[::-1]
    return flattened_data


def flatten_data(x, y):
    return flatten_array_of_arrays(x), flatten_array_of_arrays(y)


def flatten_split_data(x_train, y_train, x_val, y_val, inserting_file_separators=True, encoding=None, printing=False):
    return flatten_array_of_arrays(x_train, inserting_file_separators=inserting_file_separators, encoding=encoding,
                                   features=True, printing=printing),\
           flatten_array_of_arrays(y_train, inserting_file_separators=inserting_file_separators, encoding=encoding,
                                   features=False, printing=printing),\
           flatten_array_of_arrays(x_val, inserting_file_separators=inserting_file_separators, encoding=encoding,
                                   features=True, printing=printing),\
           flatten_array_of_arrays(y_val, inserting_file_separators=inserting_file_separators, encoding=encoding,
                                   features=False, printing=printing)


def make_dictionary_from_arrays(x, y, sources, printing=False):

    if printing:
        print(f'sources: {sources.shape}\n{sources}\n\n')

    dictionary = dict()
    i = 0
    while i < len(sources):
        # record the name of the current source file
        source = sources[i]

        # initialise lists for gathering the feature data and ground-truth data
        periodograms = list()
        ground_truth_pitches = list()

        # accumulate the file's data while the source file is the same
        while i < len(sources) and sources[i] == source:
            periodograms.insert(0, x[i])
            ground_truth_pitches.insert(0, y[i])
            i += 1

        # turn the lists into arrays and reverse the arrays
        periodograms = np.array(periodograms)[::-1]
        ground_truth_pitches = np.array(ground_truth_pitches)[::-1]

        # create the dictionary entry for that source file
        dictionary[source] = {'features': periodograms, 'ground_truth': ground_truth_pitches}

        if printing:
            print(f'source: {source}\n\n'
                  f'features: {periodograms.shape}\n{periodograms}\n\n'
                  f'ground truth: {ground_truth_pitches.shape}\n{ground_truth_pitches}\n\n')

    if printing:
        print(f'dictionary.keys(): {len(dictionary.keys())}\n{dictionary.keys()}')
    return dictionary


def flatten_dictionary(dictionary, inserting_file_separators=False, printing=False, encoded=True,
                       tracking_sources=False):
    x = list()
    y = list()
    sources = list()
    periodogram_shape = next(iter(dictionary.items()))[1]['features'].shape[1:]
    for key in dictionary.keys():
        for periodogram in dictionary[key]['features']:
            x.insert(0, periodogram)
        for label in dictionary[key]['ground_truth']:
            y.insert(0, label)
        if tracking_sources:
            for _ in dictionary[key]['features']:
                sources.insert(0, key)
        if inserting_file_separators:
            x.insert(0, np.full(periodogram_shape, -1))
            if encoded:
                y.insert(0, 89)
            else:
                y.insert(0, 'EoF')
            if tracking_sources:
                sources.insert(0, 'EoF')

    x = np.array(x)[::-1]
    y = np.array(y)[::-1]

    if tracking_sources:
        sources = np.array(sources)[::-1]

    if printing:
        print(f'number of files: {len(dictionary.keys())}')
        print(f'\nx: {x.shape}')
        print(x)
        print(f'\ny: {y.shape}')
        print(y)

    if tracking_sources:
        return x, y, sources
    else:
        return x, y


def normalise_dictionary_via_arrays(dictionary):
    x, y, sources = flatten_dictionary(dictionary, tracking_sources=True)
    x = normalise(x)
    return make_dictionary_from_arrays(x, y, sources)


def balance_dictionary(dictionary):
    x, y, sources = flatten_dictionary(dictionary, tracking_sources=True)
    x, y, sources = balance_rests(x, y, sources=sources)
    return make_dictionary_from_arrays(x, y, sources)


def split_dictionary(dictionary, n_splits=1, test_size=0.1, printing=False):
    number_of_files = len(dictionary.keys())
    x = list()
    y = list()

    for file_name in dictionary.keys():
        x.insert(0, file_name)
        y.insert(0, encode_file_name(file_name))
    x = np.array(x)[::-1]
    y = np.array(y)[::-1]

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    training_file_names = None
    validation_file_names = None
    for training_indices, validation_indices in sss.split(x, y):
        training_file_names, validation_file_names = x[training_indices], x[validation_indices]

    if printing:
        print(f'number of files: {number_of_files}\n')
        print(f'training file names: {len(training_file_names)}\n{training_file_names}\n')
        print(f'validation file names: {len(validation_file_names)}\n{validation_file_names}\n\n')

    x_train = np.empty(len(training_file_names), dtype=object)
    y_train = np.empty(len(training_file_names), dtype=object)
    x_val = np.empty(len(validation_file_names), dtype=object)
    y_val = np.empty(len(validation_file_names), dtype=object)

    for i in range(len(training_file_names)):
        x_train[i] = dictionary[training_file_names[i]]['features']
        y_train[i] = dictionary[training_file_names[i]]['ground_truth']

    for i in range(len(validation_file_names)):
        x_val[i] = dictionary[validation_file_names[i]]['features']
        y_val[i] = dictionary[validation_file_names[i]]['ground_truth']

    return x_train, y_train, x_val, y_val


def print_rnn_split_data(encoder_inputs_train, decoder_inputs_train, decoder_outputs_train,
                         encoder_inputs_val, decoder_inputs_val, decoder_outputs_val):
    print(f'encoder_inputs_train: {encoder_inputs_train.shape}\n{encoder_inputs_train}\n')
    print(f'decoder_inputs_train: {decoder_inputs_train.shape}\n{decoder_inputs_train}\n')
    print(f'decoder_outputs_train: {decoder_outputs_train.shape}\n{decoder_outputs_train}\n')
    print(f'encoder_inputs_val: {encoder_inputs_val.shape}\n{encoder_inputs_val}\n')
    print(f'decoder_inputs_val: {decoder_inputs_val.shape}\n{decoder_inputs_val}\n')
    print(f'decoder_outputs_val: {decoder_outputs_val.shape}\n{decoder_outputs_val}\n')


def save_rnn_data_arrays(save_name, encoder_inputs_train, decoder_inputs_train, decoder_outputs_train,
                         encoder_inputs_val, decoder_inputs_val, decoder_outputs_val, printing=True):

    path = f'data_arrays/{save_name}'

    try:
        mkdir(path)
    except OSError:
        print(f'Creation of the directory \"{path}\" failed.')
    else:
        if printing:
            print(f'Successfully created the directory \"{path}\".')

    np.save(f'data_arrays/{save_name}/encoder_inputs_train.npy', encoder_inputs_train)
    np.save(f'data_arrays/{save_name}/decoder_inputs_train.npy', decoder_inputs_train)
    np.save(f'data_arrays/{save_name}/decoder_outputs_train.npy', decoder_outputs_train)
    np.save(f'data_arrays/{save_name}/encoder_inputs_val.npy', encoder_inputs_val)
    np.save(f'data_arrays/{save_name}/decoder_inputs_val.npy', decoder_inputs_val)
    np.save(f'data_arrays/{save_name}/decoder_outputs_val.npy', decoder_outputs_val)

    if printing:
        print(f'Successfully saved data arrays in the directory \"{path}\".')


def load_rnn_data_arrays(save_name):
    encoder_inputs_train = np.load(f'data_arrays/{save_name}/encoder_inputs_train.npy')
    decoder_inputs_train = np.load(f'data_arrays/{save_name}/decoder_inputs_train.npy')
    decoder_outputs_train = np.load(f'data_arrays/{save_name}/decoder_outputs_train.npy')
    encoder_inputs_val = np.load(f'data_arrays/{save_name}/encoder_inputs_val.npy')
    decoder_inputs_val = np.load(f'data_arrays/{save_name}/decoder_inputs_val.npy')
    decoder_outputs_val = np.load(f'data_arrays/{save_name}/decoder_outputs_val.npy')
    return encoder_inputs_train, decoder_inputs_train, decoder_outputs_train,\
        encoder_inputs_val, decoder_inputs_val, decoder_outputs_val


def get_maximum_number_of_windows(sources, printing=False):
    maximum_number_of_windows = 0
    number_of_windows_for_current_source = 0
    i = 0
    while i < len(sources):
        current_source = sources[i]
        while i < len(sources) and sources[i] == current_source:
            number_of_windows_for_current_source += 1
            i += 1
        if number_of_windows_for_current_source > maximum_number_of_windows:
            maximum_number_of_windows = number_of_windows_for_current_source
        if printing:
            print(f'number of windows for \"{current_source}\": {number_of_windows_for_current_source:>4}')
        number_of_windows_for_current_source = 0
    if printing:
        print(f'\nmaximum number of windows: {maximum_number_of_windows}\n')
    return maximum_number_of_windows


def reshape_split_data_for_rnn(x_train, y_train, x_val, y_val, maximum_number_of_windows,
                               target_encoding='label', printing=False):
    maximum_number_of_windows += 2  # add two to account for the obligatory BoF and EoF markers
    number_of_bins = x_train[0].shape[1]

    x_train_reshaped = np.zeros(shape=(len(x_train), maximum_number_of_windows, number_of_bins))
    y_train_reshaped = np.zeros(shape=(len(y_train), maximum_number_of_windows, 1))
    x_val_reshaped = np.zeros(shape=(len(x_val), maximum_number_of_windows, number_of_bins))
    y_val_reshaped = np.zeros(shape=(len(y_val), maximum_number_of_windows, 1))

    if target_encoding == 'midi_pitch' or target_encoding == 'label':
        y_train_reshaped = y_train_reshaped.astype(int)
        y_val_reshaped = y_val_reshaped.astype(int)

    bof_label = encode_ground_truth_array(np.array(['BoF']), desired_encoding='label')[0]
    eof_label = encode_ground_truth_array(np.array(['EoF']), desired_encoding='label')[0]

    # initialise all arrays by filling them with EoF data
    x_train_reshaped[:, :] = get_eof_artificial_periodogram(number_of_bins)
    y_train_reshaped[:, :] = eof_label
    x_val_reshaped[:, :] = get_eof_artificial_periodogram(number_of_bins)
    y_val_reshaped[:, :] = eof_label

    # set the first window of every file to BoF data
    x_train_reshaped[:, 0] = get_bof_artificial_periodogram(number_of_bins)
    y_train_reshaped[:, 0] = bof_label
    x_val_reshaped[:, 0] = get_bof_artificial_periodogram(number_of_bins)
    y_val_reshaped[:, 0] = bof_label

    # for each sample in x_train, put its periodograms into x_train_reshaped
    for i in range(len(x_train)):
        x_train_i = np.array(x_train[i]).reshape(x_train[i].shape[:-1])
        # for the ith sample, from the second to the penultimate window, set the periodograms to those of x_train_i
        x_train_reshaped[i, 1:1 + len(x_train_i), :] = x_train_i
        y_train_reshaped[i, 1:1 + len(x_train_i), :] = y_train[i].reshape((y_train[i].shape[0], 1))
        assert y_train_reshaped[i, 0, 0] == BoF_LABEL_ENCODING

    for i in range(len(x_val)):
        x_val_i = np.array(x_val[i]).reshape(x_val[i].shape[:-1])
        x_val_reshaped[i, 1:1+len(x_val_i), :] = x_val_i
        y_val_reshaped[i, 1:1+len(x_val_i), :] = y_val[i].reshape((y_val[i].shape[0], 1))
        assert y_val_reshaped[i, 0, 0] == BoF_LABEL_ENCODING

    if printing:
        print(f'maximum_number_of_windows: {maximum_number_of_windows - 2} + 2 = {maximum_number_of_windows}')
        print(f'           number of bins: {number_of_bins}')
        print(f'                BoF label: {bof_label}')
        print(f'                EoF label: {eof_label}\n')
        print(f'      x_train_reshaped[0]: {x_train_reshaped[0].shape}\n{x_train_reshaped[0]}\n')
        print(f'      y_train_reshaped[0]: {y_train_reshaped[0].shape}\n{y_train_reshaped[0]}\n\n')
        print_split_data(x_train_reshaped, y_train_reshaped, x_val_reshaped, y_val_reshaped)

    return x_train_reshaped, y_train_reshaped, x_val_reshaped, y_val_reshaped


def get_decoder_outputs(decoder_inputs, printing=False):
    decoder_outputs = np.array(decoder_inputs)
    decoder_outputs[:, :-1, :] = decoder_inputs[:, 1:, :]
    if printing:
        print(f' decoder inputs: {decoder_inputs.shape}\n{decoder_inputs}\n')
        print(f'decoder outputs: {decoder_inputs.shape}\n{decoder_inputs}\n')
    return decoder_outputs


def reshape(x):
    if len(x[0].shape) == 1:  # if x is flattened
        return x.reshape(x.shape[0], x.shape[1], 1)
    elif len(x[0].shape) == 2:  # if x is not flattened
        for i in range(len(x)):
            periodograms = x[i]
            x[i] = periodograms.reshape(periodograms.shape[0], periodograms.shape[1], 1)
        return x


def load_audio_files_and_get_ground_truth(window_size, tracking_file_names=False, using_midi_bins=False,
                                          fft_strategy='scipy', wav_directory='wav_files', xml_directory='xml_files'):

    x_list = list()
    y_list = list()
    sources = list()

    # for each wav file in the data directory
    for file_name in listdir(wav_directory):
        if isfile(f'{wav_directory}/{file_name}'):

            # get the spectrogram of the audio file
            _, _, spectrogram = get_spectrogram(f'{wav_directory}/{file_name}', using_midi_bins=using_midi_bins,
                                                window_size=window_size, strategy=fft_strategy)

            # and get the ground-truth note for each periodogram in the spectrum
            file_name, _ = splitext(file_name)  # remove file extension from the filename
            ground_truth = get_monophonic_ground_truth(file_name, window_size=window_size,
                                                       wav_path=wav_directory, xml_path=xml_directory)

            # add each periodogram and its corresponding note to x_list and y_list respectively,
            # inserting data at the front of the lists for efficiency
            for i in range(len(ground_truth)):
                x_list.insert(0, spectrogram[:, i])
                y_list.insert(0, ground_truth[i])
                if tracking_file_names:
                    sources.insert(0, file_name)

    # turn the lists into arrays, reversing them to reverse the effects of inserting at the front of the lists
    x = np.array(x_list)[::-1]
    y = np.array(y_list)[::-1]
    if tracking_file_names:
        sources = np.array(sources)[::-1]

    if tracking_file_names:
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


def balance_rests(x, y, sources=None, encoding=None, printing=False, current_encoding=None):

    # sanity-test inputs
    if sources is not None:
        assert len(sources) == len(x) == len(y)
    else:
        assert len(x) == len(y)

    y_targets, y_counts = np.unique(y, return_counts=True)

    if current_encoding is None:
        average_non_rest_count = np.average(y_counts[:-1])
    else:
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


def normalise(x, strategy='k1', taking_logs=True, using_midi_bins=False,
              spectral_powers_present=True, taking_spectral_powers_logs=True, spectral_powers_strategy='max',
              first_order_differences_present=True, using_saved_maximum=False,
              printing=False):

    if using_midi_bins:
        x += np.nextafter(0.0, 1.0)

    if printing:
        print(f'Before deconstruction:\n\nx: {x.shape}\n{x}\n\n')

    spectral_powers = None
    spectral_power_differences = None
    spectral_power_differences_negatives = None
    first_order_differences = None
    first_order_differences_negatives = None

    if spectral_powers_present and first_order_differences_present:
        spectral_powers = x[:, -1, 0]
        spectral_power_differences = x[:, -1, 1]
        first_order_differences = x[:, :-1, 1]
        x = x[:, :-1, 0]
        spectral_powers = spectral_powers.reshape(-1, 1, 1)
        spectral_power_differences = spectral_power_differences.reshape(-1, 1, 1)
        first_order_differences = first_order_differences.reshape(first_order_differences.shape[0], -1, 1)
    elif spectral_powers_present:
        spectral_powers = x[:, -1]
        x = x[:, :-1]
        spectral_powers = spectral_powers.reshape(-1, 1, 1)
    elif first_order_differences_present:
        first_order_differences = x[:, :, 1]
        x = x[:, :, 0]
        first_order_differences = first_order_differences.reshape(first_order_differences.shape[0], -1, 1)
    x = x.reshape(x.shape[0], -1, 1)

    if printing:
        print(f'Before normalisation:\n')
        print(f'x: {x.shape}\n{x}\n')
        if spectral_powers_present:
            print(f'spectral_powers: {spectral_powers.shape}\n{spectral_powers}\n')
        if spectral_powers_present and first_order_differences_present:
            print(f'spectral_power_differences: {spectral_power_differences.shape}\n{spectral_power_differences}\n')
        if first_order_differences_present:
            print(f'first_order_differences: {first_order_differences.shape}\n{first_order_differences}\n\n')

    if taking_logs:
        x = np.log10(x, where=(x > 0))
        if first_order_differences_present:
            first_order_differences_negatives = np.array((first_order_differences < 0), dtype=int)
            first_order_differences = np.abs(first_order_differences)
            first_order_differences = np.log10(first_order_differences, where=(first_order_differences > 0))
    if strategy == 'k1':
        x = normalize(x, axis=1)
        if first_order_differences_present:
            first_order_differences = normalize(first_order_differences, axis=1)
    elif strategy == 'k0':
        x = normalize(x, axis=0)
        if first_order_differences_present:
            first_order_differences = normalize(first_order_differences, axis=0)
    elif strategy == 'max':
        x = x / np.amax(x)
        if first_order_differences_present:
            first_order_differences = first_order_differences / np.amax(first_order_differences)
    if first_order_differences_present:
        first_order_differences = np.concatenate((first_order_differences,
                                                  first_order_differences_negatives), axis=2)

    if spectral_powers_present:
        if taking_spectral_powers_logs:
            spectral_powers = np.log10(spectral_powers, where=(spectral_powers > 0))
            if first_order_differences_present:
                spectral_power_differences_negatives = np.array((spectral_power_differences < 0), dtype=int)
                spectral_power_differences = np.abs(spectral_power_differences)
                spectral_power_differences = np.log10(spectral_power_differences,
                                                      where=(spectral_power_differences > 0))
        if spectral_powers_strategy == 'max':
            if using_saved_maximum:
                maximum_spectral_power = np.load(f'data_arrays/maximum_spectral_power_(41773, 89, 1).npy')[0]
                spectral_powers = spectral_powers / maximum_spectral_power
            else:
                spectral_powers = spectral_powers / np.amax(spectral_powers)
            if first_order_differences_present:
                spectral_power_differences = spectral_power_differences / np.amax(spectral_power_differences)
        elif spectral_powers_strategy == 'k0':
            spectral_powers = normalize(spectral_powers, axis=0)
            if first_order_differences_present:
                spectral_power_differences = normalize(spectral_power_differences, axis=0)
        if first_order_differences_present:
            spectral_power_differences = np.concatenate((spectral_power_differences,
                                                         spectral_power_differences_negatives), axis=2)

    if printing:
        print(f'After normalisation:\n')
        if spectral_powers_present:
            print(f'spectral_powers: {spectral_powers.shape}\n{spectral_powers}\n')
        if spectral_powers_present and first_order_differences_present:
            print(f'spectral_power_differences: {spectral_power_differences.shape}\n{spectral_power_differences}\n')
        if first_order_differences_present:
            print(f'first_order_differences: {first_order_differences.shape}\n{first_order_differences}\n\n')

    if spectral_powers_present and first_order_differences_present:
        x = np.concatenate((x, spectral_powers), axis=1)
        first_order_differences = np.concatenate((first_order_differences, spectral_power_differences), axis=1)
        x = np.concatenate((x, first_order_differences), axis=2)
    elif spectral_powers_present:
        x = np.concatenate((x, spectral_powers), axis=1)
    elif first_order_differences_present:
        x = np.concatenate((x, first_order_differences), axis=2)

    if printing:
        print(f'After reconstruction\n\nx: {x.shape}\n{x}\n\n')

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

    return x_train, y_train, x_val, y_val  # n-dimensional arrays


def split_on_file_names(dictionary, printing=False):
    x_train, y_train, x_val, y_val = split_dictionary(dictionary, printing=printing)
    return x_train, y_train, x_val, y_val  # arrays of arrays


def get_data(window_size, using_midi_bins=False, wav_directory='wav_files', xml_directory='xml_files',
             splitting_on_file_name=True, adding_spectral_powers=True, adding_first_order_differences=True,
             balancing_rests=True, normalising=True, target_encoding='label', adding_file_separators=True,
             saving=False, save_name=None, printing=False, deep_printing=False):

    sources = None

    if splitting_on_file_name or adding_first_order_differences:
        x, y, sources = load_audio_files_and_get_ground_truth(window_size, using_midi_bins=using_midi_bins,
                                                              wav_directory=wav_directory, xml_directory=xml_directory,
                                                              tracking_file_names=True)
    else:
        x, y = load_audio_files_and_get_ground_truth(window_size, using_midi_bins=using_midi_bins,
                                                     wav_directory=wav_directory, xml_directory=xml_directory,
                                                     tracking_file_names=False)

    if adding_spectral_powers:
        x = add_spectral_powers(x, printing=deep_printing)

    if adding_first_order_differences:
        x, y, sources = add_first_order_differences(x, y, sources, printing=deep_printing)
    else:
        x = reshape(x)  # set up x for a single channel if it has not been so already

    if balancing_rests:
        if splitting_on_file_name:
            x, y, sources = balance_rests(x, y, sources)
        else:
            x, y = balance_rests(x, y)

    if normalising:
        x = normalise(x, spectral_powers_present=adding_spectral_powers,
                      first_order_differences_present=adding_first_order_differences, printing=deep_printing)

    y = encode(y, target_encoding=target_encoding)

    if splitting_on_file_name:
        dictionary = make_dictionary_from_arrays(x, y, sources)
        x_train, y_train, x_val, y_val = split_on_file_names(dictionary, printing=deep_printing)
        x_train, y_train, x_val, y_val = flatten_split_data(x_train, y_train, x_val, y_val, adding_file_separators,
                                                            encoding=target_encoding, printing=deep_printing)
    else:
        x_train, y_train, x_val, y_val = split(x, y)

    if printing:
        print_counts_table(y, y_train, y_val)
        print_split_data(x_train, y_train, x_val, y_val)

    if saving and save_name is not None:
        save_data_arrays(x_train, y_train, x_val, y_val, save_name=save_name)

    return x_train, y_train, x_val, y_val


def get_data_for_rnn(window_size, wav_directory='wav_files', xml_directory='xml_files', adding_spectral_powers=True,
                     balancing_rests=False, normalising=True, target_encoding='label',
                     saving=False, save_name=None, printing=False, deep_printing=False):

    x, y, sources = load_audio_files_and_get_ground_truth(window_size, wav_directory=wav_directory,
                                                          xml_directory=xml_directory, tracking_file_names=True)

    maximum_number_of_windows = get_maximum_number_of_windows(sources, printing=deep_printing)

    if adding_spectral_powers:
        x = add_spectral_powers(x, printing=deep_printing)

    if balancing_rests:
        x, y, sources = balance_rests(x, y, sources)

    if normalising:
        x = normalise(x, spectral_powers_present=adding_spectral_powers,
                      first_order_differences_present=False, printing=deep_printing)

    y = encode(y, target_encoding=target_encoding)

    dictionary = make_dictionary_from_arrays(x, y, sources)
    x_train, y_train, x_val, y_val = split_on_file_names(dictionary, printing=deep_printing)

    x_train, y_train, x_val, y_val = reshape_split_data_for_rnn(x_train, y_train, x_val, y_val,
                                                                maximum_number_of_windows,
                                                                target_encoding=target_encoding)

    encoder_inputs_train = x_train
    decoder_inputs_train = y_train
    decoder_outputs_train = get_decoder_outputs(decoder_inputs_train)
    encoder_inputs_val = x_val
    decoder_inputs_val = y_val
    decoder_outputs_val = get_decoder_outputs(decoder_inputs_val)

    if printing:
        print_rnn_split_data(encoder_inputs_train, decoder_inputs_train, decoder_outputs_train,
                             encoder_inputs_val, decoder_inputs_val, decoder_outputs_val)

    if saving and save_name is not None:
        save_rnn_data_arrays(save_name, encoder_inputs_train, decoder_inputs_train, decoder_outputs_train,
                             encoder_inputs_val, decoder_inputs_val, decoder_outputs_val)

    return encoder_inputs_train, decoder_inputs_train, decoder_outputs_train,\
        encoder_inputs_val, decoder_inputs_val, decoder_outputs_val


def main():
    get_data(25, wav_directory='wav_files', xml_directory='xml_files',
             using_midi_bins=True,
             balancing_rests=True, adding_first_order_differences=False,
             splitting_on_file_name=False, adding_file_separators=False,
             printing=True, deep_printing=True,
             saving=False, save_name='label_midi_025ms_with_powers_remove_rests_normalised')


if __name__ == '__main__':
    main()
