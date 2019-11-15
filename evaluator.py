import matplotlib.pyplot as plt
import numpy as np
from audio_processor import get_spectrogram
from scipy.interpolate import interp1d


def plot_note_graph(single_spectrum, t):
    plt.figure()
    plt.plot(t, single_spectrum)
    plt.show()


def get_maxima_times(spectrum, times, threshold, printing=False):
    peaks = list()
    maxima_times = list()
    i = 0
    while i < len(spectrum):
        # print(f'{i: >3}: {spectrum[i]}')
        if spectrum[i] >= threshold:
            peak = list()
            start_index = i
            while spectrum[i] >= threshold:
                peak.append(spectrum[i])
                i = i + 1
            # print(np.array(peak))
            peaks.append((start_index, np.array(peak)))
        i = i + 1
    for (start_index, peak) in peaks:
        maxima_times.append(times[start_index + np.argmax(peak)])
    if printing:
        print(spectrum)
        print()
        print(times)
        print()
        print(maxima_times)
    return maxima_times


def plot_smoothed(t, spectrum):
    f = interp1d(t, spectrum)
    x_new = np.linspace(0, 2.5, num=100, endpoint=True)
    f2 = interp1d(t, spectrum, kind='cubic')
    plt.plot(x_new, f(x_new), '-', x_new, f2(x_new), '--')


def main():
    spectrum, _, t, _ = get_spectrogram('scratch-wav-files/C5.wav')
    t = np.linspace(0, 2.5, 15)
    spectrum = [0.0, 0.1, 0.1, 0.5, 0.8, 0.7, 0.6, 0.5, 0.4, 0.7, 0.6, 0.4, 0.2, 0.1, 0.1, 0.0]
    # print(spectrum[len(spectrum)-1])
    # f = interp1d(t, spectrum)
    # x_new = np.linspace(0, 2.5, num=100, endpoint=True)
    # f2 = interp1d(t, spectrum, kind='cubic')
    # plt.plot(x_new, f(x_new), '-', x_new, f2(x_new), '--')
    # print(spectrum)
    # print(t)
    # plot_note_graph(spectrum[len(spectrum)-1], t)
    # plot_note_graph(spectrum, t)
    get_maxima_times(spectrum, t, 0.5, printing=True)


if __name__ == "__main__":
    main()
