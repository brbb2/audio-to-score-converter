import matplotlib.pyplot as plt
import numpy as np
import math
import wave
import scipy.signal


window_size_parameters = {
    25: {'nperseg': 2048, 'noverlap': 1024},
    50: {'nperseg': 4096, 'noverlap': 2048},
    100: {'nperseg': 8192, 'noverlap': 4096},
    200: {'nperseg': 16384, 'noverlap': 8192},
}


def get_window_parameters(window_size):
    return window_size_parameters[window_size]['nperseg'], window_size_parameters[window_size]['noverlap']


def get_audio_signal(wav_file):
    wav = wave.open(wav_file, 'r')
    signal = wav.readframes(-1)
    return np.frombuffer(signal, dtype=np.int32)


def print_wav_details(wav_file):
    wav = wave.open(wav_file, 'r')
    signal = get_audio_signal(wav_file)
    frame_rate = wav.getframerate()
    duration = len(signal) / frame_rate

    print(f'frame rate = {frame_rate} frames per second')
    print(f'size = {signal.size} frames')
    print(f'duration = {duration:.2f} seconds')


def plot_audio_signal(wav_file, seconds=True, showing=True):
    wav = wave.open(wav_file, 'r')
    signal = wav.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int32)

    plt.figure()
    plt.title(f'Audio Signal of \"{wav_file}\"')

    if seconds:
        frame_rate = wav.getframerate()
        duration = len(signal)/frame_rate
        xs = np.linspace(0, duration, num=len(signal))
        plt.plot(xs, signal)
        plt.xlabel('time (seconds)')
    else:
        plt.plot(signal)
        plt.xlabel('frame')
    plt.ylim(-0.1, 0.1)
    plt.xlim(2.2, 2.201)

    plt.ylabel('Displacement')

    if showing:
        plt.show()


def print_spectrogram(spectrum, frequencies, t):
    print("spectrum:", spectrum.shape)
    print(spectrum)
    print()
    print("frequencies:", frequencies.shape)
    print(frequencies)
    print()
    print("column mid-points:", t.shape)
    print(t)


def get_spectrogram_scipy(wav_file, window='hamming', nperseg=4096, noverlap=2048, printing=False,
                          using_midi_bins=False):
    wav = wave.open(wav_file, 'r')
    signal = wav.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int32)
    if printing:
        print(wav.getframerate())
    f, t, sxx = scipy.signal.spectrogram(signal, wav.getframerate(),
                                         window=window, nperseg=nperseg, noverlap=noverlap)

    if using_midi_bins:
        sxx = merge_frequencies(sxx, f)

    if printing:
        print_spectrogram(sxx, f, t)
    return f, t, sxx


def get_spectrogram_pyplot(wav_file, nfft=4096, noverlap=2048, seconds=True, printing=False):
    wav = wave.open(wav_file, 'r')
    signal = wav.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int32)
    if seconds:
        spectrum, frequencies, t, im = plt.specgram(signal, NFFT=nfft, Fs=wav.getframerate(), noverlap=noverlap)
    else:
        spectrum, frequencies, t, im = plt.specgram(signal, NFFT=nfft, Fs=1, noverlap=noverlap)
    if printing:
        print_spectrogram(spectrum, frequencies, t)
    return spectrum, frequencies, t, im


def get_spectrogram_numpy(wav_file, window_size=0.2, sampling_frequency=44100, printing=False):
    wav = wave.open(wav_file, 'r')
    signal = wav.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int32)
    duration = signal.size / float(sampling_frequency)
    spectrogram = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(signal.shape[-1])
    frequencies = frequencies * sampling_frequency
    # window_duration = 0.05
    # samples_per_window = math.ceil(window_duration * sampling_frequency)
    samples_per_window = math.ceil(window_size * sampling_frequency)
    number_of_windows = math.ceil(signal.size / samples_per_window)
    windows = np.empty((number_of_windows, samples_per_window))
    # print(windows.shape)
    padded_signal = np.zeros(number_of_windows * samples_per_window)
    padded_signal[:signal.shape[0]] = signal
    spectrogram = np.empty((samples_per_window, number_of_windows))
    for i in range(number_of_windows):
        window = padded_signal[i*samples_per_window:(i+1)*samples_per_window]
        windows[i, :] = window
    for i in range(number_of_windows):
        # print(f'{windows[i, :].size} {windows[i, :]}')
        window = windows[i, :]
        spectrogram[:, i] = np.fft.fft(window).real

    print(spectrogram.shape)
    print(spectrogram)
    frequencies = np.fft.fftfreq(samples_per_window) * sampling_frequency
    spectrogram = np.positive(spectrogram)
    for i in range(number_of_windows):
        if i == 30:
            continue
            spectrogram[:, i] = np.ones(samples_per_window)
        else:
            spectrogram[:, i] = np.zeros(samples_per_window)

    # plt.figure()
    # plt.pcolormesh(spectrogram)
    # plt.show()

    if printing:
        print(f'samples per window: {samples_per_window} samples per window')
        print(f'number of windows: {number_of_windows} windows')
        print(f'{duration:.2f} seconds')
        print(signal.shape)
        print(frequencies)
        print("spectrum:")
        print(windows)
        print(windows.shape)
    return spectrogram


def get_spectrogram(wav_file, strategy='scipy', window='hamming', nperseg=4096, noverlap=2048,
                    printing=False, using_midi_bins=False, seconds=True):
    if strategy == 'scipy':
        frequencies, times, spectrogram = get_spectrogram_scipy(wav_file, window=window, nperseg=nperseg,
                                                                noverlap=noverlap, printing=printing,
                                                                using_midi_bins=using_midi_bins)
    elif strategy == 'pyplot':
        spectrogram, frequencies, times, _ = get_spectrogram_pyplot(wav_file, nfft=nperseg, noverlap=noverlap,
                                                                    seconds=seconds, printing=printing)

    return frequencies, times, spectrogram


def plot_spectrogram_scipy(wav_file, window='hamming', nperseg=4096, noverlap=2048,
                           plotting_logged=True, returning_logged=False, midi=False, seconds=True,
                           showing=True, printing=False):

    f, t, sxx = get_spectrogram_scipy(wav_file, window=window, nperseg=nperseg, noverlap=noverlap,
                                      using_midi_bins=midi, printing=printing)

    plt.figure()
    plt.title(f'scipy.signal.spectrogram\n\"{wav_file}\"')

    if seconds:
        x = t
        plt.xlabel('time (seconds)')
    else:
        x = range(len(t))
        plt.xlabel('frame')

    if midi:
        plt.ylabel('MIDI pitch')
        y = range(21, 109)
    else:
        plt.ylabel('frequency (Hz)')
        plt.ylim(0, 5000)
        y = f

    if plotting_logged:
        plt.pcolormesh(x, y, np.log10(sxx))
    else:
        plt.pcolormesh(x, y, sxx)

    if showing:
        plt.show()

    if returning_logged:
        sxx = np.log10(sxx)

    return f, t, sxx


def plot_spectrogram_pyplot(wav_file, nfft=4096, noverlap=2048, seconds=True, showing=True):
    wav = wave.open(wav_file, 'r')
    signal = wav.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int32)

    plt.figure()
    plt.title(f'matplotlib.pyplot.specgram\n\"{wav_file}\"')

    if seconds:
        spectrum, frequencies, times, im = plt.specgram(signal, NFFT=nfft, Fs=wav.getframerate(), noverlap=noverlap)
        plt.xlabel('time (seconds)')
    else:
        spectrum, frequencies, times, im = plt.specgram(signal, NFFT=nfft, Fs=1, noverlap=noverlap)
        plt.xlabel('frame')

    plt.ylabel('frequency (Hz)')
    # plt.ylim(0, 5000)

    if showing:
        plt.show()

    return spectrum, frequencies, times, im


def get_periodograms(wav_file, library='scipy'):
    if library == 'scipy':
        _, _, spectrum = get_spectrogram_scipy(wav_file)
    else:
        spectrum, _, _, _ = get_spectrogram_pyplot(wav_file)
    return np.swapaxes(spectrum, 0, 1)


def merge_frequencies(spectrum, frequencies, printing=False):
    frequency_bound = 440 * 2 ** (-95 / 24.0)

    midi_spectrum = np.full((88, spectrum.shape[1]), 1e-5)  # fill with small number to avoid log(0)

    if printing:
        print(f'  shape of original spectrum: {str(spectrum.shape):>10}')
        print(f'shape of MIDI-pitch spectrum: {str(midi_spectrum.shape):>10}')
        print(f'    starting frequency bound: {frequency_bound:7.2f} Hz')
        print()

    i = 0
    midi_pitch = 21
    while midi_pitch < 109:
        contributors = 0
        while i < frequencies.size and frequencies[i] < frequency_bound:
            contributors += 1
            if printing:
                print(f'adding frequency bin {frequencies[i]:7.2f} Hz to MIDI-pitch bin {midi_pitch:>3}')
            midi_spectrum[midi_pitch - 21, :] += spectrum[i, :]
            i += 1
        if contributors > 1:
            midi_spectrum[midi_pitch - 21, :] = midi_spectrum[midi_pitch - 21, :] / float(contributors)
        frequency_bound *= 2 ** (1 / 12.0)
        midi_pitch += 1

    return midi_spectrum


def main():
    note = 'C4'
    example = 3
    test_wav = f'wav_files/single_{note}_{example}.wav'

    # plot_audio_signal(test_wav)

    # plot_spectrogram_scipy(test_wav, showing=False)
    # plot_spectrogram_scipy(test_wav, plotting_logged=False, midi=True, showing=False)
    plot_spectrogram_pyplot(test_wav, showing=False)
    plot_spectrogram_pyplot(f'wav_files/wav_files_no_reverb/single_{note}_{example}.wav', showing=False)
    plt.show()

    # f, t, sxx = get_spectrogram_scipy(test_wav)
    # print(sxx.shape)
    # print(sxx)


if __name__ == "__main__":
    main()
