import matplotlib.pyplot as plt
import numpy as np
import wave


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


def main():
    # print_wav_details('scratch-wav-files/A4.wav')
    # plot_audio_signal('scratch-wav-files/test.wav')
    plot_spectrogram('scratch-wav-files/Scarborough Fair.wav')


if __name__ == "__main__":
    main()
