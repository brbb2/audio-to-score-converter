from audio_processor import *
import unittest


class TestAudioProcessor(unittest.TestCase):

    def setUp(self):
        self.test_wav = 'wav_files/single_A0_0.wav'

    def test_assert_not_using_midi_bins_for_pyplot(self):
        self.assertRaises(AssertionError, plot_spectrogram, self.test_wav, strategy='pyplot', using_midi_bins=True)

    def test_merge_frequencies_output_shape(self):
        frequencies, _, spectrogram = get_spectrogram(self.test_wav)
        midi_spectrum = merge_frequency_bins_into_midi_bins(spectrogram, frequencies)
        print(f'original spectrum shape: {str(spectrogram.shape):>10}')
        print(f'    midi spectrum shape: {str(midi_spectrum.shape):>10}')
        self.assertEqual(midi_spectrum.shape, (88, spectrogram.shape[1]))


if __name__ == '__main__':
    unittest.main()
