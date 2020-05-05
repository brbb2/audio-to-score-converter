from data_processor import *
from keras.utils import normalize


def print_data(x, y):
    print(f'x: {x.shape}')
    print(x)
    print(f'\ny: {y.shape}')
    print(y)


def shuffle_data(x, y):
    np.random.seed(42)
    np.random.shuffle(x)
    np.random.seed(42)
    np.random.shuffle(y)
    return x, y


def get_data_periodograms_flattened(midi_bins=False, window_size=25,
                                    wav_path='wav_files', strategy='scipy', tracking_sources=False):

    x_list = list()
    y_list = list()
    sources = list()

    # for each wav file in the data directory
    for file_name in listdir(wav_path):
        if isfile(f'wav_files/{file_name}'):

            # get the spectrogram of the audio file
            _, _, spectrogram = get_spectrogram(f'{wav_path}/{file_name}', using_midi_bins=midi_bins,
                                                window_size=window_size, strategy=strategy)

            # and get the ground-truth note for each periodogram in the spectrum
            file_name, _ = splitext(file_name)  # remove file extension from the filename
            ground_truth = get_monophonic_ground_truth(file_name, window_size, wav_path=wav_path)

            # add each periodogram and its corresponding note to x_list and y_list respectively,
            # inserting data at the front of the lists for efficiency
            for i in range(len(ground_truth)):
                x_list.insert(0, spectrogram[:, i])
                y_list.insert(0, ground_truth[i])
                if tracking_sources:
                    sources.insert(0, file_name)

    # turn the lists into arrays, reversing them to reverse the effects of inserting at the front of the lists
    x = np.array(x_list)[::-1]
    y = np.array(y_list)[::-1]
    if tracking_sources:
        sources = np.array(sources)[::-1]

    if tracking_sources:
        return x, y, sources
    else:
        return x, y


def get_data_periodograms_not_flattened(midi_bins=False, window_size=25,
                                        wav_path='wav_files', strategy='scipy'):

    x_list = list()
    y_list = list()

    # for each wav file in the data directory
    for file_name in listdir(wav_path):
        if isfile(f'{wav_path}/{file_name}'):

            # get the spectrogram of the audio file
            _, _, spectrogram = get_spectrogram(f'{wav_path}/{file_name}', using_midi_bins=midi_bins,
                                                window_size=window_size, strategy=strategy)

            # and get the ground-truth note for each periodogram in the spectrum
            file_name, _ = splitext(file_name)  # remove file extension from the filename
            ground_truth = get_monophonic_ground_truth(file_name, window_size, wav_path=wav_path)

            # add the spectrogram data and its ground-truth pitches to x_list and y_list respectively,
            # inserting data at the front of the lists for efficiency
            x_list.insert(0, np.swapaxes(spectrogram, 0, 1))
            y_list.insert(0, ground_truth)

    # turn the lists into arrays, reversing them to reverse the effects of inserting at the front of the lists
    x = np.array(x_list)[::-1]
    y = np.array(y_list)[::-1]

    return x, y


def get_spectral_powers(x, printing=False):
    x_squared = np.apply_along_axis(np.square, 1, x)
    spectral_powers = np.apply_along_axis(np.sum, 1, x_squared)
    spectral_powers = spectral_powers.reshape(spectral_powers.shape[0], 1)
    if printing:
        print(f'              x.shape: {x.shape}')
        print(f'spectral_powers.shape: {spectral_powers.shape}')
    return spectral_powers


def test_flatten_array_of_arrays():
    x, y = get_data(25, splitting_on_file_name=True)
    x = flatten_array_of_arrays(x)
    y = flatten_array_of_arrays(y)
    print(f'x: {x.shape}\n{x}\n\ny: {y.shape}\n{y}')
    x2, y2 = get_data(25, splitting_on_file_name=False)
    for i in range(len(y)):
        if y[i] != y2[i]:
            print(f'False: y1[{i}] != y2[{i}]')
            return
    else:
        print(f'flatten_array_of_arrays(x), flatten_array_of_arrays(y) == get_data(flattening=True, shuffling=False)')


def test_normalisations():
    x, y = get_data_periodograms_flattened()
    print(f'y: {y.shape}\n{y}\n\n')
    print(f'x: {x.shape}\n{x}\n\n')
    x_norm_0 = normalize(x, axis=0)
    print(f'keras.utils.normalise(x, axis=0): {x_norm_0.shape}\n{x_norm_0}\n\n')
    x_norm_1 = normalize(x, axis=1)
    print(f'keras.utils.normalize(x, axis=1): {x_norm_1.shape}\n{x_norm_1}\n\n')
    x_norm_max = x / np.amax(x)
    print(f'x / np.amax(x): {x_norm_max.shape}\n{x_norm_max}\n\n')
    x_norm_log_max = np.log10(x) / np.amax(x)
    print(f'np.log10(x) / np.amax(x): {x_norm_log_max.shape}\n{x_norm_log_max}')


def test_get_spectral_powers():
    x, _ = get_data_periodograms_flattened()
    spectral_powers = get_spectral_powers(x)
    print(f'    spectral powers: {spectral_powers.shape}\n{spectral_powers}\n')
    print(f'spectral powers max: {np.amax(spectral_powers)}')
    print(f'spectral powers min: {np.amin(spectral_powers)}\n')
    spectral_powers_k0_norm = normalize(spectral_powers, axis=0)
    print(f'      k0 normalised: {spectral_powers_k0_norm.shape}\n{spectral_powers_k0_norm}\n')
    print(f'  k0 normalised max: {np.amax(spectral_powers_k0_norm)}')
    print(f'  k0 normalised min: {np.amin(spectral_powers_k0_norm)}\n')
    spectral_powers_max_norm = spectral_powers / np.amax(spectral_powers)
    print(f'     max normalised: {spectral_powers_max_norm.shape}\n{spectral_powers_max_norm}\n')
    print(f' max normalised max: {np.amax(spectral_powers_max_norm)}')
    print(f' max normalised min: {np.amin(spectral_powers_max_norm)}')


def test_get_data_periodograms_not_flattened():
    x, y = get_data_periodograms_not_flattened()
    print_data(x, y)


def test_reshape():
    x, _ = get_data_periodograms_flattened()
    x = reshape(x)
    print(f'    x_flattened_reshaped: {x.shape}\n{x}\n')

    x, _ = get_data_periodograms_not_flattened()
    x = reshape(x)
    print(f'x_not_flattened_reshaped: {x.shape}\n{x}')


def test_get_data_periodograms_flattened():
    x, y, sources = get_data_periodograms_flattened(tracking_sources=True)
    print(f'x: {x.shape}\n{x}\n\ny: {y.shape}\n{y}\n\nsources: {sources.shape}\n{sources}')


def test_balance_rests():
    x, y = load_audio_files_and_get_ground_truth(25, wav_directory='wav_files_simple', xml_directory='xml_files_simple',
                                                 using_midi_bins=True)
    print('before balancing rests:\n')
    _, y_train, _, y_val = split(x, y)
    print_counts_table(y, y_train, y_val)

    x, y = balance_rests(x, y)
    print('\n\nafter balancing rests:\n')
    _, y_train, _, y_val = split(x, y)
    print_counts_table(y, y_train, y_val)


def main():
    test_balance_rests()


if __name__ == '__main__':
    main()
