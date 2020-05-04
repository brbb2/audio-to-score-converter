from neural_network_trainer import *
from data_processor import *
from keras.utils import normalize


def test_flatten_array_of_arrays():
    x, y = get_data(flattening=False, shuffling=False)
    x = flatten_array_of_arrays(x)
    y = flatten_array_of_arrays(y)
    print(f'x: {x.shape}\n{x}\n\ny: {y.shape}\n{y}')
    x2, y2 = get_data(flattening=True, shuffling=False)
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


def test_add_first_order_difference():
    x, y = get_data_periodograms_not_flattened()
    x_reshaped = reshape(x)

    x, y = add_first_order_difference(x, y)
    x_reshaped, _ = add_first_order_difference(x_reshaped, y)

    x = flatten_array_of_arrays(x)
    y = flatten_array_of_arrays(y)
    x_reshaped = flatten_array_of_arrays(x_reshaped)

    print(f'x: {x_reshaped.shape}\n{x_reshaped}\n')
    print_data(x, y)


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
