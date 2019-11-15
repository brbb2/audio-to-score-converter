import matplotlib.pyplot as plt
import numpy as np
import wave
from sklearn.decomposition import NMF


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


def plot_audio_signal(wav_file, seconds=True):
    wav = wave.open(wav_file, 'r')
    signal = wav.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int32)

    plt.figure(figsize=(7, 5))

    if seconds:
        frame_rate = wav.getframerate()
        duration = len(signal)/frame_rate
        xs = np.linspace(0, duration, num=len(signal))
        plt.plot(xs, signal)
        plt.xlabel('time (s)')
    else:
        plt.plot(signal)
        plt.xlabel('frame')

    plt.title('Audio Signal')
    plt.ylabel('y Label')
    plt.show()


def get_spectrogram(wav_file, nfft=16384, noverlap=8192, seconds=True, printing=False):
    wav = wave.open(wav_file, 'r')
    signal = wav.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int32)
    plt.figure(figsize=(6, 5))
    if seconds:
        spectrum, frequencies, t, im = plt.specgram(signal, NFFT=nfft, Fs=wav.getframerate(), noverlap=noverlap)
        plt.xlabel('time (s)')
    else:
        spectrum, frequencies, t, im = plt.specgram(signal, NFFT=nfft, Fs=1, noverlap=noverlap)
        plt.xlabel('frame')
    if printing:
        print("spectrum:", spectrum.shape)
        print(spectrum)
        print()
        print("frequencies:", frequencies.shape)
        print(frequencies)
        print()
        print("column mid-points:", t.shape)
        print(t)
    return spectrum, frequencies, t, im


def plot_spectrogram(wav_file, nfft=16384, noverlap=8192, seconds=True):
    wav = wave.open(wav_file, 'r')
    signal = wav.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int32)
    plt.figure(figsize=(6, 5))
    if seconds:
        (_, frequencies, bins, _) = plt.specgram(signal, NFFT=nfft, Fs=wav.getframerate(), noverlap=noverlap)
        plt.xlabel('time (s)')
    else:
        (_, frequencies, bins, _) = plt.specgram(signal, NFFT=nfft, Fs=1, noverlap=noverlap)
        plt.xlabel('frame')
    # print(frequencies)
    # print(bins)
    plt.ylim(0, 3520)
    plt.title(f'Spectrogram of \'{wav_file}\'')
    plt.ylabel('frequency (Hz)')
    plt.show()


def factorise_matrix(matrix, printing=False):
    model = NMF(n_components=2, init='random', random_state=0)
    w = model.fit_transform(matrix)
    h = model.components_
    if printing:
        print('W =')
        print(w)
        print('\nH =')
        print(h)
    return w, h


def main():
    # print_wav_details('scratch-wav-files/A4.wav')
    # plot_audio_signal('scratch-wav-files/A4.wav')
    get_spectrogram('scratch-wav-files/C5.wav', printing=True)
    # plot_spectrogram('scratch-wav-files/Scarborough Fair.wav')
    # factorise_matrix(np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]]), printing=True)


if __name__ == "__main__":
    main()
