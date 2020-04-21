import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import wave


window_size_parameters = {
    25: {'nperseg': 2048, 'noverlap': 1024},
    50: {'nperseg': 4096, 'noverlap': 2048},
    100: {'nperseg': 8192, 'noverlap': 4096},
    200: {'nperseg': 16384, 'noverlap': 8192},
}


def get_window_parameters(window_size):
    return window_size_parameters[window_size]['nperseg'], window_size_parameters[window_size]['noverlap']


def get_precise_window_duration_in_seconds(approximate_window_size, printing=False):
    nperseg, noverlap = get_window_parameters(approximate_window_size)
    sampling_frequency = 44100.0
    window_duration = (nperseg - noverlap) / sampling_frequency
    if printing:
        print(f'window duration: {window_duration:6.4f} seconds')
    return window_duration


def print_wav_details(wav_file):
    signal, frame_rate = get_audio_signal(wav_file, returning_frame_rate=True)
    duration = len(signal) / frame_rate

    print(f'frame rate: {frame_rate} frames per second')
    print(f'      size: {signal.size} frames')
    print(f'  duration: {duration:.2f} seconds')


def get_audio_signal(wav_file, returning_frame_rate=False):

    # read the wav file data into a numpy array of integers
    wav = wave.open(wav_file, 'r')
    signal = wav.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int32)

    if returning_frame_rate:
        frame_rate = wav.getframerate()
        wav.close()
        return signal, frame_rate
    else:
        wav.close()
        return signal


def plot_audio_signal(wav_file, plotting_seconds_on_the_x_axis=True, showing=True,
                      x_min=None, x_max=None, y_min=None, y_max=None):

    signal, frame_rate = get_audio_signal(wav_file, returning_frame_rate=True)

    plt.figure()
    plt.title(f'Audio Signal of \"{wav_file}\"')

    if plotting_seconds_on_the_x_axis:
        duration = len(signal) / frame_rate
        xs = np.linspace(0, duration, num=len(signal))
        plt.plot(xs, signal)
        plt.xlabel('time (seconds)')
    else:
        plt.plot(signal)
        plt.xlabel('frame')

    if x_max is not None:
        if x_min is not None:
            plt.xlim(0, x_max)
        else:
            plt.xlim(x_min, x_max)

    if y_max is not None:
        if y_min is not None:
            plt.ylim(0, y_max)
        else:
            plt.ylim(y_min, y_max)

    plt.ylabel('displacement')

    if showing:
        plt.show()


def print_spectrogram_data(frequencies, times, spectrogram):
    print(f'    spectrogram: {spectrogram.shape}\n{spectrogram}\n\n'
          f'    frequencies: {frequencies.shape}\n{frequencies}\n\n'
          f'mid-point times: {times.shape}\n{times}')


def get_spectrogram_scipy(wav_file, window_size=25, using_midi_bins=False, window_type='hamming', printing=False):

    # get the corresponding parameters for the specified window size
    nperseg, noverlap = get_window_parameters(window_size)

    signal, frame_rate = get_audio_signal(wav_file, returning_frame_rate=True)

    # compute the spectrogram using the 'scipy' library
    frequencies, times, spectrogram = scipy.signal.spectrogram(signal, frame_rate,
                                                               window=window_type,
                                                               nperseg=nperseg, noverlap=noverlap)

    if using_midi_bins:
        spectrogram = merge_frequency_bins_into_midi_bins(spectrogram, frequencies)

    if printing:
        print(f'{wav_file}:\n\nframe rate: {frame_rate}\n')
        print_spectrogram_data(spectrogram, frequencies, times)

    return frequencies, times, spectrogram


def get_spectrogram_pyplot(wav_file, window_size=25, plotting_seconds_on_the_x_axis=True, printing=False):

    # get the corresponding parameters for the specified window size
    nperseg, noverlap = get_window_parameters(window_size)

    signal, frame_rate = get_audio_signal(wav_file, returning_frame_rate=True)

    # compute the spectrogram using the 'pyplot' library
    if plotting_seconds_on_the_x_axis:
        spectrogram, frequencies, times, _ = plt.specgram(signal, NFFT=nperseg, noverlap=noverlap, Fs=frame_rate)
    else:
        spectrogram, frequencies, times, _ = plt.specgram(signal, NFFT=nperseg, noverlap=noverlap, Fs=1)

    if printing:
        print_spectrogram_data(frequencies, times, spectrogram)

    return frequencies, times, spectrogram


def get_spectrogram(wav_file, window_size=25, strategy='scipy', window_type='hamming',
                    printing=False, using_midi_bins=False, plotting_seconds_on_the_x_axis=True):
    
    frequencies = None
    times = None
    spectrogram = None
    
    if strategy == 'scipy':
        frequencies, times, spectrogram = get_spectrogram_scipy(wav_file, window_size=window_size,
                                                                window_type=window_type, printing=printing,
                                                                using_midi_bins=using_midi_bins)
    elif strategy == 'pyplot':
        frequencies, times, spectrogram = get_spectrogram_pyplot(wav_file, window_size=window_size,
                                                                 plotting_seconds_on_the_x_axis=
                                                                 plotting_seconds_on_the_x_axis,
                                                                 printing=printing)

    return frequencies, times, spectrogram


def plot_spectrogram_scipy(wav_file, window_size=25, window_type='hamming',
                           plotting_logged=False, returning_logged=False, using_midi_bins=False,
                           plotting_seconds_on_x_axis=True, y_max=None, showing=True, printing=False):

    frequencies, times, spectrogram = get_spectrogram_scipy(wav_file, window_size=window_size, window_type=window_type,
                                                            using_midi_bins=using_midi_bins, printing=printing)

    plt.figure()
    plt.title(f'scipy.signal.spectrogram\n\"{wav_file}\"')

    if plotting_seconds_on_x_axis:
        x = times
        plt.xlabel('time (seconds)')
    else:
        x = range(len(times))
        plt.xlabel('frame')

    if using_midi_bins:
        plt.ylabel('MIDI pitch')
        y = range(21, 109)
    else:
        plt.ylabel('frequency (Hz)')
        if y_max is not None:
            plt.ylim(0, y_max)
        y = frequencies

    if plotting_logged:
        plt.pcolormesh(x, y, np.log10(spectrogram))
    else:
        plt.pcolormesh(x, y, spectrogram)

    if showing:
        plt.show()

    if returning_logged:
        spectrogram = np.log10(spectrogram)

    return frequencies, times, spectrogram


def plot_spectrogram_pyplot(wav_file, window_size=25, plotting_seconds_on_the_x_axis=True, y_max=None, showing=True):

    # get the corresponding parameters for the specified window size
    nperseg, noverlap = get_window_parameters(window_size)

    signal, frame_rate = get_audio_signal(wav_file, returning_frame_rate=True)

    plt.figure()
    plt.title(f'matplotlib.pyplot.specgram\n\"{wav_file}\"')

    if plotting_seconds_on_the_x_axis:
        spectrogram, frequencies, times, _ = plt.specgram(signal, NFFT=nperseg, noverlap=noverlap, Fs=frame_rate)
        plt.xlabel('time (seconds)')
    else:
        spectrogram, frequencies, times, _ = plt.specgram(signal, NFFT=nperseg, noverlap=noverlap, Fs=1)
        plt.xlabel('frame')

    plt.ylabel('frequency (Hz)')
    
    if y_max is not None:
        plt.ylim(0, y_max)

    if showing:
        plt.show()

    return frequencies, times, spectrogram


def plot_spectrogram(wav_file, window_size=25, strategy='pyplot', window_type='hamming',
                     plotting_logged=True, returning_logged=False, using_midi_bins=False,
                     plotting_seconds_on_x_axis=True, y_max=None, showing=True, printing=False):

    assert not (using_midi_bins and strategy == 'pyplot')

    if strategy == 'scipy':
        plot_spectrogram_scipy(wav_file, window_size=window_size, window_type=window_type,
                               plotting_logged=plotting_logged, returning_logged=returning_logged,
                               using_midi_bins=using_midi_bins, plotting_seconds_on_x_axis=plotting_seconds_on_x_axis,
                               y_max=y_max, showing=showing, printing=printing)
    elif strategy == 'pyplot':
        plot_spectrogram_pyplot(wav_file, window_size=window_size, y_max=y_max, showing=showing)


def get_periodograms(wav_file, window_size=25, strategy='scipy'):
    spectrogram = None
    if strategy == 'scipy':
        _, _, spectrogram = get_spectrogram_scipy(wav_file, window_size=window_size)
    elif strategy == 'pyplot':
        _, _, spectrogram = get_spectrogram_pyplot(wav_file, window_size=window_size)
    # swap the axis to return an array of periodograms
    return np.swapaxes(spectrogram, 0, 1)


def merge_frequency_bins_into_midi_bins(spectrogram, frequencies, printing=False):

    # initialise the first frequency bound to the geometric mid-point of A0 and B-0
    frequency_bound = 440 * 2 ** (-95 / 24.0)

    midi_spectrogram = np.full((88, spectrogram.shape[1]), 0.0)

    if printing:
        print(f'  shape of original spectrum: {str(spectrogram.shape):>10}')
        print(f'shape of MIDI-pitch spectrum: {str(midi_spectrogram.shape):>10}')
        print(f'    starting frequency bound: {frequency_bound:7.2f} Hz')
        print()

    i = 0
    midi_pitch = 21
    while midi_pitch < 109:
        contributors = 0
        # while a frequency bin corresponds to the current MIDI bin, accumulate its power into that bin
        while i < len(frequencies) and frequencies[i] < frequency_bound:
            contributors += 1
            if printing:
                print(f'adding frequency bin {frequencies[i]:7.2f} Hz to MIDI-pitch bin {midi_pitch:>3}')
            midi_spectrogram[midi_pitch - 21, :] += spectrogram[i, :]
            i += 1

        # take the average power across the relevant frequency bins by dividing by the number of contributing bins
        if contributors > 1:
            midi_spectrogram[midi_pitch - 21, :] = midi_spectrogram[midi_pitch - 21, :] / float(contributors)

        # go to the next frequency upper-bound for the next MIDI pitch
        frequency_bound *= 2 ** (1 / 12.0)
        midi_pitch += 1

    return midi_spectrogram


def main():
    note = 'A6'
    example = 7
    test_wav = f'wav_files/single_{note}_{example}.wav'

    plot_spectrogram(test_wav, strategy='scipy', using_midi_bins=False)


if __name__ == "__main__":
    main()
