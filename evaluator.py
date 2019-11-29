import matplotlib.pyplot as plt
import numpy as np
import random
from audio_processor import get_spectrogram
from scipy.interpolate import interp1d
from neural_network_trainer import load_model


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

    spectrum, _, times, _ = get_spectrogram(f'wav_files/{wav_name}.wav')
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
    model = load_model('baseline')
    spectrum, _, times, _ = get_spectrogram(f'wav_files/single_D4_0.wav')
    print(spectrum.shape)
    p = np.zeros(spectrum.shape[1])
    a = np.zeros(spectrum.shape[1])
    for i in range(spectrum.shape[1]):
        periodogram = spectrum[:, i].reshape(1, 8193, 1)
        pitch_probabilities = model.predict(periodogram)[0]
        a[i] = np.argmax(pitch_probabilities)
        print(pitch_probabilities)
        p[i] = pitch_probabilities[10]
    print()
    print(a)


if __name__ == "__main__":
    main()
