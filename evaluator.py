import numpy as np
import matplotlib.pyplot as plt
from keras.utils import normalize
from scipy.interpolate import interp1d
from data_processor import add_spectral_powers, normalise
from audio_processor import get_spectrogram, plot_spectrogram
from ground_truth_converter import get_monophonic_ground_truth
from neural_network_trainer import load_model, load_data_arrays, get_data_file_names
from encoder import interpret_one_hot, get_one_hot_midi_pitch_index, get_note_name, get_pitch_array, REST_MIDI_ENCODING


def predict_each_window_of_wav_file(file_name, wav_path='wav_files', adding_spectral_powers=True, normalising=True,
                                    window_size=50, model_name=None):

    if wav_path is None:
        wav_file_full_path = f'{file_name}.wav'
    else:
        wav_file_full_path = f'{wav_path}/{file_name}.wav'

    # get the spectrogram of the file and swap the axes to get an array of periodograms
    _, _, spectrogram = get_spectrogram(wav_file_full_path, window_size=window_size)
    periodograms = np.swapaxes(spectrogram, 0, 1)
    if adding_spectral_powers:
        periodograms = add_spectral_powers(periodograms)
    periodograms = periodograms.reshape((periodograms.shape[0], periodograms.shape[1], 1))

    if normalising:
        periodograms = normalise(periodograms, spectral_powers_present=adding_spectral_powers,
                                 first_order_differences_present=False)

    # load the specified model and use it to predict the pitch at each window
    model = load_model(model_name)
    probabilities = model.predict(periodograms)
    predictions = np.empty(len(probabilities), dtype=object)
    for i in range(len(probabilities)):
        predictions[i] = interpret_one_hot(probabilities[i], encoding=None)


def create_comparison_text_file(file_name, model_name, window_size=50, wav_path='wav_files', xml_path='xml_files',
                                adding_spectral_powers=True, normalising=True,
                                save_name=None, printing=False):

    if save_name is None:
        save_name = file_name

    if wav_path is None:
        wav_file_full_path = f'{file_name}.wav'
    else:
        wav_file_full_path = f'{wav_path}/{file_name}.wav'

    # get the ground-truth pitch for the file
    ground_truth = get_monophonic_ground_truth(file_name, wav_path=wav_path, xml_path=xml_path, window_size=window_size)

    # get the spectrogram of the file and swap the axes to get an array of periodograms
    _, _, spectrogram = get_spectrogram(wav_file_full_path, window_size=window_size)
    periodograms = np.swapaxes(spectrogram, 0, 1)
    if adding_spectral_powers:
        periodograms = add_spectral_powers(periodograms)
    periodograms = periodograms.reshape((periodograms.shape[0], periodograms.shape[1], 1))

    if normalising:
        periodograms = normalise(periodograms, spectral_powers_present=adding_spectral_powers,
                                 first_order_differences_present=False)

    # load the specified model and use it to predict the pitch at each window
    model = load_model(model_name)
    probabilities = model.predict(periodograms)
    predictions = np.empty(len(probabilities), dtype=object)
    for i in range(len(probabilities)):
        predictions[i] = interpret_one_hot(probabilities[i], encoding=None)

    # write the ground truth pitches and pitch predictions to a text file
    f = open(f'txt_files/{save_name}.txt', 'w')
    f.write('        time step:   ')
    for time_step in range(len(ground_truth)):
        f.write(f'{time_step:<5}')
    f.write('\n')
    f.write('     ground truth:   ')
    for pitch in ground_truth:
        f.write(f'{pitch:<5}')
    f.write('\n')
    f.write('model predictions:   ')
    for pitch in predictions:
        f.write(f'{pitch:<5}')
    f.close()

    if printing:
        print(spectrogram.shape)
        print(spectrogram)
        print()
        print(predictions.shape)
        print(predictions)
        print()
        print(ground_truth.shape)
        print(ground_truth)


def create_all_comparison_text_files(model_name, window_size=50, printing=True, deep_printing=False):
    data_file_names = get_data_file_names()
    for data_file_name in data_file_names:
        create_comparison_text_file(data_file_name, model_name, window_size=window_size,
                                    printing=deep_printing)
        if printing:
            print(f'Created comparison text file for \"{data_file_name}\".')


def plot_note_graph(single_spectrum, t):
    plt.figure()
    plt.plot(t, single_spectrum)
    plt.show()


def get_maxima_times(single_frequency_spectrum, times, threshold, printing=False):
    peaks = list()
    maxima_times = list()
    i = 0
    while i < len(single_frequency_spectrum):
        # print(f'{i: >3}: {spectrum[i]}')
        if single_frequency_spectrum[i] >= threshold:
            peak = list()
            start_index = i
            while i < len(single_frequency_spectrum) and single_frequency_spectrum[i] >= threshold:
                peak.append(single_frequency_spectrum[i])
                i = i + 1
            # print(np.array(peak))
            peaks.append((start_index, np.array(peak)))
        i = i + 1
    for (start_index, peak) in peaks:
        maxima_times.append(times[start_index + np.argmax(peak)])
    if printing:
        print(single_frequency_spectrum)
        print()
        print(times)
        print()
        print(maxima_times)
    return np.array(maxima_times)


def plot_smoothed(t, spectrogram):
    f = interp1d(t, spectrogram)
    x_new = np.linspace(0, 2.5, num=100, endpoint=True)
    f2 = interp1d(t, spectrogram, kind='cubic')
    plt.plot(x_new, f(x_new), '-', x_new, f2(x_new), '--')


def plot_wav_prediction(note, example, model_name, method='scipy', printing=False,
                        plotting_spectrogram=False, showing=True):
    wav_file = f'wav_files/single_{note}_{example}.wav'
    model = load_model(model_name)
    if method == 'scipy':
        if plotting_spectrogram:
            _, times, spectrogram = plot_spectrogram(wav_file, strategy='scipy', showing=False)
        else:
            _, times, spectrogram = get_spectrogram(wav_file, strategy='scipy')
    else:
        if plotting_spectrogram:
            _, times, spectrogram = plot_spectrogram(wav_file, strategy='pyplot', showing=False)
        else:
            _, times, spectrogram = get_spectrogram(wav_file, strategy='pyplot')
    if printing:
        print(spectrogram.shape)
    midi_pitch_predictions = np.zeros(spectrogram.shape[1])
    for i in range(spectrogram.shape[1]):
        periodogram = spectrogram[:, i].reshape(1, spectrogram.shape[0], 1)
        pitch_probabilities = model.predict(periodogram)[0]
        midi_pitch_predictions[i] = interpret_one_hot(pitch_probabilities)
        if printing:
            print(midi_pitch_predictions[i])
            print(pitch_probabilities)

    plt.figure()
    plt.title(f'Prediction of \"{wav_file}\"\nby model \"{model_name}\"')
    plt.plot(times, midi_pitch_predictions)
    plt.xlabel('time (seconds)')
    plt.ylabel('MIDI pitch')
    plt.ylim(-2, 109)
    if showing:
        plt.show()


def plot_pitch_probability(note, example, midi_pitch, model_name, strategy='scipy', printing=False,
                           showing=True, plotting_new_figure=True, plotting_spectrogram=False,
                           plotting_legend=False, encoding=None):

    wav_file = f'wav_files/single_{note}_{example}.wav'
    model = load_model(model_name)
    pitch_index = get_one_hot_midi_pitch_index(midi_pitch)

    if strategy == 'scipy':
        if plotting_spectrogram:
            _, times, spectrogram = plot_spectrogram(wav_file, strategy='scipy', showing=False)
        else:
            _, times, spectrogram = get_spectrogram(wav_file, strategy='scipy')
    else:
        if plotting_spectrogram:
            _, times, spectrogram = plot_spectrogram(wav_file, strategy='pyplot', showing=False)
        else:
            _, times, spectrogram = get_spectrogram(wav_file, strategy='pyplot')

    spectrogram = normalize(spectrogram, axis=0)
    pitch_probability = np.zeros(spectrogram.shape[1])
    for i in range(spectrogram.shape[1]):
        periodogram = spectrogram[:, i]
        # periodogram = normalize(periodogram, axis=0)[0]
        model_input = periodogram.reshape(1, periodogram.shape[0], 1)
        pitch_probabilities = model.predict(model_input)[0]
        pitch_probability[i] = pitch_probabilities[pitch_index]
        if printing:
            print(f'At timestep {i}, the probability of MIDI pitch {midi_pitch} is {pitch_probability[i]}.')
            print(f'\nmodel_input: {model_input.shape}')
            print(model_input)
            print(f'\npitch_prediction: {pitch_probabilities.shape}')
            print(pitch_probabilities)

    if plotting_new_figure:
        plt.figure()
        plt.title(f'Probability of MIDI pitch {midi_pitch}\nin \"{wav_file}\"')
        plt.xlabel('time (seconds)')
        plt.ylabel('probability')
        plt.ylim(-0.05, 1.05)

    if plotting_legend:
        if encoding is None:
            label = get_note_name(midi_pitch)
        else:
            label = midi_pitch
        plt.plot(times, pitch_probability, label=label)
    else:
        plt.plot(times, pitch_probability)

    if showing:
        plt.show()


def plot_all_pitch_probabilities(note, example, model_name, start=21, end=108, method='scipy', showing=True):
    plot_pitch_probability(note, example, REST_MIDI_ENCODING, 'new3', plotting_new_figure=False,
                           plotting_legend=True, showing=False)
    for i in range(start, end+1):
        plot_pitch_probability(note, example, i, model_name, strategy=method, plotting_new_figure=False,
                               plotting_legend=True, showing=False)
    plt.legend()
    if showing:
        plt.show()


def plot_pitch_accuracies(data_version, model_name, title=None, alpha=1,
                          showing=True, printing=False, plotting_new_figure=True):

    # load the specified validation data and the specified model
    _, _, x_val, y_val = load_data_arrays(data_version)
    model = load_model(model_name)

    # use the model to give the pitch probabilities for every periodogram in the validation set
    pitch_probabilities = model.predict(x_val)

    # initialise counts to 0 for all pitches
    counts = np.zeros(89, dtype=int)
    correct_counts = np.zeros(89, dtype=int)

    # for every periodogram in the validation set,
    # get the most likely pitch and determine whether it matches the ground truth,
    # then update the counts accordingly
    for i in range(len(x_val)):
        prediction = np.argmax(pitch_probabilities[i])
        ground_truth = y_val[i]
        counts[ground_truth] += 1
        if prediction == ground_truth:
            correct_counts[ground_truth] += 1
        if printing:
            print(f'ground truth: {ground_truth:>3}    prediction: {prediction:>3}')

    pitch_accuracies = correct_counts / counts.astype(float)

    if printing:
        print()
        print(counts)
        print()
        print(correct_counts)
        print()
        print(pitch_accuracies)
        print()
        print(f'overall accuracy: {np.sum(correct_counts) / float(np.sum(counts))}')

    pitches = get_pitch_array()

    if plotting_new_figure:
        plt.figure()
        if title is None:
            plt.title(f'Prediction accuracy for each MIDI pitch\nwith model \"{model_name}\"')
        else:
            plt.title(title)
        plt.xlabel('MIDI pitch')
        plt.ylabel('Accuracy')

    plt.bar(pitches, pitch_accuracies, alpha=alpha, label=model_name)

    if showing:
        plt.show()


def main():
    # plot_pitch_accuracies('debugged', 'label_freq_050ms_remove_rests_10_powers_log_k1_norm_dense_debugged',
    #                       alpha=0.5, showing=True, plotting_new_figure=True)
    plot_pitch_accuracies('label_freq_025ms_with_powers_remove_rests_normalised',
                          'label_freq_025ms_with_powers_remove_rests_log_k1_normalised_using_dropout_midi_model',
                          title=f'Prediction accuracy for each MIDI pitch\nwith best model',
                          alpha=0.5, showing=True, plotting_new_figure=True)
    # note_name = 'G4'
    # example = 3
    # create_comparison_text_file(f'single_{note_name}_{example}',
    #                             'label_freq_050ms_remove_rests_10_powers_log_k1_norm_dense_debugged',
    #                             wav_path='wav_files_simple', xml_path='xml_files_simple',
    #                             save_name=f'single_{note_name}_{example}_debugged')


if __name__ == "__main__":
    main()
