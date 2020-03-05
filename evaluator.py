import matplotlib.pyplot as plt
import numpy as np
import random
from audio_processor import get_spectrogram_scipy, get_spectrogram_pyplot
from audio_processor import plot_spectrogram_scipy, plot_spectrogram_pyplot
from ground_truth_converter import get_monophonic_ground_truth
from scipy.interpolate import interp1d
from neural_network_trainer import load_model, load_model_and_get_predictions, load_data_arrays, get_data_file_names
from keras.utils import normalize
import midi_manager


def create_comparison_text_file(file_name, model_name, nperseg=4096, noverlap=2048, printing=False):
    _, _, spectrum = get_spectrogram_scipy(f'wav_files/{file_name}.wav', nperseg=nperseg, noverlap=noverlap)
    periodograms = np.swapaxes(spectrum, 0, 1)
    periodograms = periodograms.reshape(periodograms.shape[0], periodograms.shape[1], 1)
    ground_truth = get_monophonic_ground_truth(f'wav_files/{file_name}.wav', f'xml_files/{file_name}.musicxml',
                                               encoding=None, nperseg=nperseg, noverlap=noverlap)
    model = load_model(model_name)
    probabilities = model.predict(periodograms)
    predictions = np.empty(len(probabilities), dtype=object)
    for i in range(len(probabilities)):
        predictions[i] = midi_manager.interpret_one_hot(probabilities[i], encoding=None)

    # write the ground truth pitches and pitch predictions to a text file
    f = open(f'txt_files/{file_name}.txt', 'w')
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
        print(spectrum.shape)
        print(spectrum)
        print()
        print(predictions.shape)
        print(predictions)
        print()
        print(ground_truth.shape)
        print(ground_truth)


def create_all_comparison_text_files(model_name, nperseg, noverlap, printing=True, deep_printing=False):
    data_file_names = get_data_file_names()
    for data_file_name in data_file_names:
        create_comparison_text_file(data_file_name, model_name, nperseg=nperseg, noverlap=noverlap,
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


def get_maxima_times_for_all_pitches(spectrum, times, threshold, printing=False):
    note_predictions = np.empty(len(spectrum), dtype=object)
    for i in range(len(spectrum)):
        single_frequency_spectrum = spectrum[i]
        maxima_times = get_maxima_times(single_frequency_spectrum, times, threshold, printing=printing)
        note_predictions[i] = maxima_times
    return note_predictions


def get_note_predictions(pitches, spectrum, times, threshold):
    maxima_times_for_all_pitches = get_maxima_times_for_all_pitches(spectrum, times, threshold)
    note_predictions = np.empty(len(pitches), dtype=object)
    for i in range(len(pitches)):
        note_predictions[i] = (pitches[i], maxima_times_for_all_pitches[i])
    return note_predictions


def separate_notes(note_predictions):
    separated_notes = list()
    for note_prediction in note_predictions:
        for onset in note_prediction[1]:
            separated_notes.insert(0, (note_prediction[0], onset))
    return np.array(separated_notes)


def get_sorted_notes(pitches, spectrum, times, threshold, printing=False):
    note_predictions = get_note_predictions(pitches, spectrum, times, threshold)
    separated_notes = separate_notes(note_predictions)
    separated_notes_sorted = sorted(separated_notes, key=lambda separated_note: separated_note[1])
    if printing:
        print(note_predictions)
        print()
        print(separated_notes_sorted)
    return separated_notes_sorted


def plot_smoothed(t, spectrum):
    f = interp1d(t, spectrum)
    x_new = np.linspace(0, 2.5, num=100, endpoint=True)
    f2 = interp1d(t, spectrum, kind='cubic')
    plt.plot(x_new, f(x_new), '-', x_new, f2(x_new), '--')


def get_pitch_probabilities(wav_name, model, number_of_pitches=89, plotting=False, printing=False):

    spectrum, _, times, _ = get_spectrogram_pyplot(f'wav_files/{wav_name}.wav')
    all_pitch_activations = np.empty(number_of_pitches, dtype=object)

    if plotting:
        plt.figure()
        plt.title('Pitch Activations')
        plt.xlabel('Time (s)')
        plt.ylabel('Probability')
        plt.ylim(0, 1.05)

    for pitch in range(89):
        pitch_activation = np.zeros(len(times))
        for i in range(len(times)):
            periodogram = spectrum[:, i].reshape(1, 8193, 1)
            pitch_probabilities = model.predict(periodogram)[0]
            pitch_probability = pitch_probabilities[pitch]
            pitch_activation[i] = pitch_probability
        if plotting and (pitch == 88 or pitch == 44):
            plt.plot(times, pitch_activation, label=pitch)
        all_pitch_activations[pitch] = pitch_activation

        if printing:
            print(f'periodogram.shape: {periodogram.shape}\n')
            print('periodogram:')
            print(periodogram)
            print()
            print('pitch_probabilities:')
            print(pitch_probabilities)
            print()
            print(f'pitch_{pitch}_probability: {pitch_probability}')

    if plotting:
        plt.legend()
        plt.show()

    return all_pitch_activations


def plot_wav_prediction(note, example, model_name, method='scipy', printing=False,
                        plotting_spectrogram=False, showing=True):
    wav_file = f'wav_files/single_{note}_{example}.wav'
    model = load_model(model_name)
    if method == 'scipy':
        if plotting_spectrogram:
            _, times, spectrum = plot_spectrogram_scipy(wav_file, showing=False)
        else:
            _, times, spectrum = get_spectrogram_scipy(wav_file)
    else:
        if plotting_spectrogram:
            spectrum, _, times, _ = plot_spectrogram_pyplot(wav_file, showing=False)
        else:
            spectrum, _, times, _ = get_spectrogram_pyplot(wav_file)
    if printing:
        print(spectrum.shape)
    midi_pitch_predictions = np.zeros(spectrum.shape[1])
    for i in range(spectrum.shape[1]):
        periodogram = spectrum[:, i].reshape(1, spectrum.shape[0], 1)
        pitch_probabilities = model.predict(periodogram)[0]
        midi_pitch_predictions[i] = midi_manager.interpret_one_hot(pitch_probabilities)
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


def plot_pitch_probability(note, example, midi_pitch, model_name, method='scipy', printing=False,
                           showing=True, plotting_new_figure=True, plotting_spectrogram=False,
                           plotting_legend=False, encoding=None):

    wav_file = f'wav_files/single_{note}_{example}.wav'
    model = load_model(model_name)
    pitch_index = midi_manager.get_one_hot_midi_pitch_index(midi_pitch)

    if method == 'scipy':
        if plotting_spectrogram:
            _, times, spectrum = plot_spectrogram_scipy(wav_file, showing=False)
        else:
            _, times, spectrum = get_spectrogram_scipy(wav_file)
    else:
        if plotting_spectrogram:
            spectrum, _, times, _ = plot_spectrogram_pyplot(wav_file, showing=False)
        else:
            spectrum, _, times, _ = get_spectrogram_pyplot(wav_file)
    spectrum = normalize(spectrum, axis=0)
    pitch_probability = np.zeros(spectrum.shape[1])
    for i in range(spectrum.shape[1]):
        periodogram = spectrum[:, i]
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
            label = midi_manager.get_note_name(midi_pitch)
        else:
            label = midi_pitch
        plt.plot(times, pitch_probability, label=label)
    else:
        plt.plot(times, pitch_probability)

    if showing:
        plt.show()


def plot_all_pitch_probabilities(note, example, model_name, start=21, end=108, method='scipy', showing=True):
    plot_pitch_probability(note, example, midi_manager.REST_ENCODING, 'new3', plotting_new_figure=False,
                           plotting_legend=True, showing=False)
    for i in range(start, end+1):
        plot_pitch_probability(note, example, i, model_name, method=method, plotting_new_figure=False,
                               plotting_legend=True, showing=False)
    plt.legend()
    if showing:
        plt.show()


def plot_pitch_accuracies(data_version, model_name, alpha=1, showing=True, printing=False, plotting_new_figure=True):
    _, _, x_val, y_val = load_data_arrays(data_version)
    model = load_model(model_name)
    predictions = model.predict(x_val)
    counts = np.zeros(89, dtype=int)
    correct_counts = np.zeros(89, dtype=int)

    for i in range(len(y_val)):
        prediction = np.argmax(predictions[i])
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

    pitches = midi_manager.get_pitch_array()

    if plotting_new_figure:
        plt.figure()
        plt.title(f'Prediction accuracy for each MIDI pitch\nwith model \"{model_name}\"')
        plt.xlabel('MIDI pitch')
        plt.ylabel('Accuracy')

    plt.bar(pitches, pitch_accuracies, alpha=alpha, label=model_name)

    if showing:
        plt.show()


def main():
    # spectrum, _, t, _ = get_spectrogram('scratch-wav-files/C5.wav')
    # number_of_pitches = 7
    # number_of_time_steps = 10
    # t = np.linspace(0, 2.5, number_of_time_steps)
    # pitches = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # single_frequency_spectrum = [0.0, 0.1, 0.1, 0.5, 0.8, 0.7, 0.6, 0.5, 0.4, 0.7, 0.6, 0.4, 0.2, 0.1, 0.1, 0.0]
    # spectrum = np.empty(number_of_pitches, dtype=object)
    # for i in range(number_of_pitches):
    #     single_frequency_spectrum = np.zeros(number_of_time_steps)
    #     for j in range(number_of_time_steps):
    #         single_frequency_spectrum[j] = random.uniform(0, 1)
    #     spectrum[i] = single_frequency_spectrum
    # print(single_frequency_spectrum[len(single_frequency_spectrum)-1])
    # f = interp1d(t, single_frequency_spectrum)
    # x_new = np.linspace(0, 2.5, num=100, endpoint=True)
    # f2 = interp1d(t, single_frequency_spectrum, kind='cubic')
    # plt.plot(x_new, f(x_new), '-', x_new, f2(x_new), '--')
    # print(single_frequency_spectrum)
    # print(spectrum)
    # print(t)
    # plot_note_graph(single_frequency_spectrum[len(single_frequency_spectrum)-1], t)
    # plot_note_graph(single_frequency_spectrum, t)
    # get_maxima_times(single_frequency_spectrum, t, 0.5, printing=True)
    # print(get_maxima_times_for_all_pitches(spectrum, t, 0.7))
    # note_predictions = get_note_predictions(pitches, spectrum, t, 0.7)
    # print(note_predictions)
    # print()
    # sorted(note_predictions, key=lambda note_prediction: note_prediction[1][0])
    # print(note_predictions)
    # separated_notes = separate_notes(note_predictions)
    # separated_notes_sorted = get_sorted_notes(pitches, spectrum, t, 0.7)
    # print(separated_notes_sorted)
    # print()
    # plot_pitch_accuracies('label_freq_050ms', 'label_freq_050ms_remove_rests', alpha=0.5, showing=False,
    #                       plotting_new_figure=False)
    plot_pitch_accuracies('label_midi_050ms_remove_rests_10_powers_log_k1_norm_dense_4', 'label_midi_050ms_remove_rests_10_powers_log_k1_norm_dense_4', alpha=0.5, showing=False, plotting_new_figure=False)
    # plot_wav_prediction('C4', 0, 'one_hot_freq_050ms', plotting_spectrogram=True, method='pyplot')
    # plot_pitch_probability('C4', 3, 60, 'new3', showing=False)
    # plot_all_pitch_probabilities('C4', 3, 'new3')
    # create_comparison_text_file('single_C4_3', 'new3')
    plt.title(f'Pitch accuracies for each MIDI pitch\nwith models \"label_freq_050ms\" and \"label_freq_050ms_remove_rests\"')
    plt.xlabel('MIDI pitch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
