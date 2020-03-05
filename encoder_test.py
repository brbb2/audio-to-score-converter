import numpy as np
import unittest
import encoder


class TestEncoder(unittest.TestCase):

    def setUp(self):
        self.test_wav = 'wav_files/single_A0_0.wav'

    def test_get_midi_pitch_on_rest(self):
        midi_pitch = encoder.get_midi_pitch('rest')
        self.assertEqual(midi_pitch, encoder.REST_MIDI_ENCODING)

    def test_get_midi_pitch_on_A0(self):
        midi_pitch = encoder.get_midi_pitch('A0')
        self.assertEqual(midi_pitch, 21)

    def test_get_midi_pitch_on_F2(self):
        midi_pitch = encoder.get_midi_pitch('F2')
        self.assertEqual(midi_pitch, 41)

    def test_get_midi_pitch_on_Ab3(self):
        midi_pitch = encoder.get_midi_pitch('B-3')
        self.assertEqual(midi_pitch, 58)

    def test_get_midi_pitch_on_Cb4(self):
        midi_pitch = encoder.get_midi_pitch('C-4')
        self.assertEqual(midi_pitch, 59)

    def test_get_midi_pitch_on_Es4(self):
        midi_pitch = encoder.get_midi_pitch('E#4')
        self.assertEqual(midi_pitch, 65)

    def test_get_midi_pitch_on_C8(self):
        midi_pitch = encoder.get_midi_pitch('C8')
        self.assertEqual(midi_pitch, 108)

    def test_get_note_name(self):
        print()
        note_name = encoder.get_note_name(encoder.REST_MIDI_ENCODING)
        print(note_name, end=', ')
        self.assertEqual(note_name, 'rest')
        for i in range(21, 109):
            note_name = encoder.get_note_name(i)
            if i < 108:
                print(note_name, end=', ')
            else:
                print(note_name)
            if i == 21:
                self.assertEqual(note_name, 'A0')
            elif i == 24:
                self.assertEqual(note_name, 'C1')
            elif i == 51:
                self.assertEqual(note_name, 'E-3')

    def test_one_hot_encode_midi_pitch_on_rest(self):
        one_hot_encoding = encoder.one_hot_encode_midi_pitch(encoder.REST_MIDI_ENCODING)
        self.assertEqual(np.argmax(one_hot_encoding), 0)

    def test_one_hot_encode_midi_pitch_on_A0(self):
        one_hot_encoding = encoder.one_hot_encode_midi_pitch(21)
        self.assertEqual(np.argmax(one_hot_encoding), 1)

    def test_one_hot_encode_midi_pitch_on_C8(self):
        one_hot_encoding = encoder.one_hot_encode_midi_pitch(108)
        self.assertEqual(np.argmax(one_hot_encoding), 88)

    def test_interpret_one_hot_on_rest(self):
        midi_pitch_input = encoder.REST_MIDI_ENCODING
        one_hot = encoder.one_hot_encode_midi_pitch(midi_pitch_input)
        print()
        midi_pitch_output = encoder.interpret_one_hot(one_hot, printing=True)
        print(f'midi_pitch_input: {midi_pitch_input:>3}    midi_pitch_output: {midi_pitch_output:>3}')
        self.assertEqual(midi_pitch_input, midi_pitch_output)

    def test_interpret_one_hot_on_A0(self):
        midi_pitch_input = 21
        one_hot = encoder.one_hot_encode_midi_pitch(midi_pitch_input)
        print()
        midi_pitch_output = encoder.interpret_one_hot(one_hot, printing=True)
        print(f'midi_pitch_input: {midi_pitch_input:>3}    midi_pitch_output: {midi_pitch_output:>3}')
        self.assertEqual(midi_pitch_input, midi_pitch_output)

    def test_interpret_one_hot_on_putting_rests_last(self):
        midi_pitch_input = encoder.REST_MIDI_ENCODING
        one_hot = encoder.one_hot_encode_midi_pitch(midi_pitch_input, putting_rests_last=True)
        print()
        midi_pitch_output = encoder.interpret_one_hot(one_hot, putting_rests_last=True, printing=True)
        print(f'midi_pitch_input: {midi_pitch_input:>3}    midi_pitch_output: {midi_pitch_output:>3}')
        self.assertEqual(midi_pitch_input, midi_pitch_output)

    def test_interpret_one_hot_on_putting_lows_last(self):
        midi_pitch_input = 60
        one_hot = encoder.one_hot_encode_midi_pitch(midi_pitch_input, low_last=True)
        print()
        midi_pitch_output = encoder.interpret_one_hot(one_hot, low_last=True, printing=True)
        print(f'midi_pitch_input: {midi_pitch_input:>3}    midi_pitch_output: {midi_pitch_output:>3}')
        self.assertEqual(midi_pitch_input, midi_pitch_output)

    def test_midi_pitch_encode_note_name(self):
        test_array = np.array(['BoF', 'rest', 'C4', 'A-1', 'F#7', 'EoF'])
        encoded_array = encoder.encode_ground_truth_array(test_array,
                                                          current_encoding=None, desired_encoding='midi_pitch')
        print(f'\n\ntest array: {test_array.shape}\n{test_array}\n')
        print(f'encoded array: {encoded_array.shape}\n{encoded_array}')
        self.assertEqual(encoded_array[0], encoder.BoF_MIDI_ENCODING)
        self.assertEqual(encoded_array[1], encoder.REST_MIDI_ENCODING)
        self.assertEqual(encoded_array[2], 60)
        self.assertEqual(encoded_array[3], 32)
        self.assertEqual(encoded_array[4], 102)
        self.assertEqual(encoded_array[5], encoder.EoF_MIDI_ENCODING)

    def test_label_encode_note_name(self):
        test_array = np.array(['BoF', 'rest', 'C4', 'A-1', 'F#7', 'EoF'])
        encoded_array = encoder.encode_ground_truth_array(test_array,
                                                          current_encoding=None, desired_encoding='label')
        print(f'\n\ntest array: {test_array.shape}\n{test_array}\n')
        print(f'encoded array: {encoded_array.shape}\n{encoded_array}')
        self.assertEqual(encoded_array[0], 89)
        self.assertEqual(encoded_array[1], 0)
        self.assertEqual(encoded_array[2], 40)
        self.assertEqual(encoded_array[3], 12)
        self.assertEqual(encoded_array[4], 82)
        self.assertEqual(encoded_array[5], 90)

    def test_one_hot_encode_note_name(self):
        test_array = np.array(['rest', 'C4', 'A-1', 'F#7'])
        encoded_array = encoder.encode_ground_truth_array(test_array,
                                                          current_encoding=None, desired_encoding='one_hot',
                                                          for_rnn=False)
        print(f'\n\ntest array: {test_array.shape}\n{test_array}\n')
        print(f'encoded array: {encoded_array.shape} of {encoded_array[0].shape}\n{encoded_array}')
        self.assertEqual(len(encoded_array[0]), 89)
        self.assertEqual(encoded_array[0][0], 1)
        self.assertEqual(encoded_array[1][40], 1)
        self.assertEqual(encoded_array[2][12], 1)
        self.assertEqual(encoded_array[3][82], 1)

    def test_one_hot_encode_note_name_for_rnn(self):
        test_array = np.array(['BoF', 'rest', 'C4', 'A-1', 'F#7', 'EoF'])
        encoded_array = encoder.encode_ground_truth_array(test_array,
                                                          current_encoding=None, desired_encoding='one_hot',
                                                          for_rnn=True)
        print(f'\n\ntest array: {test_array.shape}\n{test_array}\n')
        print(f'encoded array: {encoded_array.shape} of {encoded_array[0].shape}\n{encoded_array}')
        self.assertEqual(len(encoded_array[0]), 91)
        self.assertEqual(encoded_array[0][89], 1)
        self.assertEqual(encoded_array[1][0], 1)
        self.assertEqual(encoded_array[2][40], 1)
        self.assertEqual(encoded_array[3][12], 1)
        self.assertEqual(encoded_array[4][82], 1)
        self.assertEqual(encoded_array[5][90], 1)

    def test_decode_midi_pitch(self):
        encoded_array = np.array([encoder.BoF_MIDI_ENCODING, encoder.REST_MIDI_ENCODING, 60, 32, 102,
                                  encoder.EoF_MIDI_ENCODING])
        decoded_array = encoder.encode_ground_truth_array(encoded_array,
                                                          current_encoding='midi_pitch', desired_encoding=None)
        print(f'\n\nencoded array: {encoded_array.shape}\n{encoded_array}\n')
        print(f'decoded array: {decoded_array.shape}\n{decoded_array}')
        self.assertEqual(decoded_array[0], 'BoF')
        self.assertEqual(decoded_array[1], 'rest')
        self.assertEqual(decoded_array[2], 'C4')
        self.assertEqual(decoded_array[3], 'A-1')
        self.assertEqual(decoded_array[4], 'F#7')
        self.assertEqual(decoded_array[5], 'EoF')

    def test_label_encode_midi_pitch(self):
        test_array = np.array([encoder.BoF_MIDI_ENCODING, encoder.REST_MIDI_ENCODING, 60, 32, 102,
                               encoder.EoF_MIDI_ENCODING])
        encoded_array = encoder.encode_ground_truth_array(test_array,
                                                          current_encoding='midi_pitch', desired_encoding='label')
        print(f'\n\ntest array: {test_array.shape}\n{test_array}\n')
        print(f'encoded array: {encoded_array.shape}\n{encoded_array}')
        self.assertEqual(encoded_array[0], 89)
        self.assertEqual(encoded_array[1], 0)
        self.assertEqual(encoded_array[2], 40)
        self.assertEqual(encoded_array[3], 12)
        self.assertEqual(encoded_array[4], 82)
        self.assertEqual(encoded_array[5], 90)

    def test_one_hot_encode_midi_pitch(self):
        test_array = np.array([encoder.REST_MIDI_ENCODING, 60, 32, 102])
        encoded_array = encoder.encode_ground_truth_array(test_array,
                                                          current_encoding='midi_pitch', desired_encoding='one_hot',
                                                          for_rnn=False)
        print(f'\n\ntest array: {test_array.shape}\n{test_array}\n')
        print(f'encoded array: {encoded_array.shape} of {encoded_array[0].shape}\n{encoded_array}')
        self.assertEqual(len(encoded_array[0]), 89)
        self.assertEqual(encoded_array[0][0], 1)
        self.assertEqual(encoded_array[1][40], 1)
        self.assertEqual(encoded_array[2][12], 1)
        self.assertEqual(encoded_array[3][82], 1)

    def test_one_hot_encode_midi_pitch_for_rnn(self):
        test_array = np.array([encoder.BoF_MIDI_ENCODING, encoder.REST_MIDI_ENCODING, 60, 32, 102,
                               encoder.EoF_MIDI_ENCODING])
        encoded_array = encoder.encode_ground_truth_array(test_array,
                                                          current_encoding='midi_pitch', desired_encoding='one_hot',
                                                          for_rnn=True)
        print(f'\n\ntest array: {test_array.shape}\n{test_array}\n')
        print(f'encoded array: {encoded_array.shape} of {encoded_array[0].shape}\n{encoded_array}')
        self.assertEqual(len(encoded_array[0]), 91)
        self.assertEqual(encoded_array[0][89], 1)
        self.assertEqual(encoded_array[1][0], 1)
        self.assertEqual(encoded_array[2][40], 1)
        self.assertEqual(encoded_array[3][12], 1)
        self.assertEqual(encoded_array[4][82], 1)
        self.assertEqual(encoded_array[5][90], 1)

    def test_decode_label(self):
        encoded_array = np.array([89, 0, 40, 12, 82, 90])
        decoded_array = encoder.encode_ground_truth_array(encoded_array,
                                                          current_encoding='label', desired_encoding=None)
        print(f'\n\nencoded array: {encoded_array.shape}\n{encoded_array}\n')
        print(f'decoded array: {decoded_array.shape}\n{decoded_array}')
        self.assertEqual(decoded_array[0], 'BoF')
        self.assertEqual(decoded_array[1], 'rest')
        self.assertEqual(decoded_array[2], 'C4')
        self.assertEqual(decoded_array[3], 'A-1')
        self.assertEqual(decoded_array[4], 'F#7')
        self.assertEqual(decoded_array[5], 'EoF')

    def test_midi_encode_code_label(self):
        encoded_array = np.array([89, 0, 40, 12, 82, 90])
        decoded_array = encoder.encode_ground_truth_array(encoded_array,
                                                          current_encoding='label', desired_encoding='midi_pitch')
        print(f'\n\nencoded array: {encoded_array.shape}\n{encoded_array}\n')
        print(f'decoded array: {decoded_array.shape}\n{decoded_array}')
        self.assertEqual(decoded_array[0], encoder.BoF_MIDI_ENCODING)
        self.assertEqual(decoded_array[1], encoder.REST_MIDI_ENCODING)
        self.assertEqual(decoded_array[2], 60)
        self.assertEqual(decoded_array[3], 32)
        self.assertEqual(decoded_array[4], 102)
        self.assertEqual(decoded_array[5], encoder.EoF_MIDI_ENCODING)

    def test_one_hot_encode_label(self):
        test_array = np.array([0, 40, 12, 82])
        encoded_array = encoder.encode_ground_truth_array(test_array,
                                                          current_encoding='label', desired_encoding='one_hot',
                                                          for_rnn=False)
        print(f'\n\ntest array: {test_array.shape}\n{test_array}\n')
        print(f'encoded array: {encoded_array.shape} of {encoded_array[0].shape}\n{encoded_array}')
        self.assertEqual(len(encoded_array[0]), 89)
        self.assertEqual(encoded_array[0][0], 1)
        self.assertEqual(encoded_array[1][40], 1)
        self.assertEqual(encoded_array[2][12], 1)
        self.assertEqual(encoded_array[3][82], 1)

    def test_one_hot_encode_label_for_rnn(self):
        test_array = np.array([89, 0, 40, 12, 82, 90])
        encoded_array = encoder.encode_ground_truth_array(test_array,
                                                          current_encoding='label', desired_encoding='one_hot',
                                                          for_rnn=True)
        print(f'\n\ntest array: {test_array.shape}\n{test_array}\n')
        print(f'encoded array: {encoded_array.shape} of {encoded_array[0].shape}\n{encoded_array}')
        self.assertEqual(len(encoded_array[0]), 91)
        self.assertEqual(encoded_array[0][89], 1)
        self.assertEqual(encoded_array[1][0], 1)
        self.assertEqual(encoded_array[2][40], 1)
        self.assertEqual(encoded_array[3][12], 1)
        self.assertEqual(encoded_array[4][82], 1)
        self.assertEqual(encoded_array[5][90], 1)


if __name__ == '__main__':
    unittest.main()
