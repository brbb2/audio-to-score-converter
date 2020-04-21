from predictor import *
import unittest


class TestPredictor(unittest.TestCase):

    def setUp(self):
        self.test_file_name = 'Pomp_and_Circumstance_Theme'

    def test_prediction_starts_at_time_zero(self):
        predicted_notes = predict(self.test_file_name, threshold=0.65)
        self.assertAlmostEqual(predicted_notes[0][1], 0.0)

    def test_get_pitches_of_notes(self):
        notes_of_a_major = [('A3', 0, 1), ('B3', 1, 2), ('C#4', 2, 3), ('D4', 3, 4),
                            ('E4', 4, 5), ('F#4', 5, 6), ('G#4', 6, 7)]
        pitches_of_a_major = get_pitches_of_notes(notes_of_a_major)
        self.assertEqual(pitches_of_a_major, ['A3', 'B3', 'C#4', 'D4', 'E4', 'F#4', 'G#4'])

    def test_infer_key_signature_a_major(self):
        notes_of_a_major = [('A3', 0, 1), ('B3', 1, 2), ('C#4', 2, 3), ('D4', 3, 4),
                            ('E4', 4, 5), ('F#4', 5, 6), ('G#4', 6, 7)]
        predicted_key_signature = infer_key_signature(notes_of_a_major)
        self.assertEqual(predicted_key_signature, 3)

    def test_infer_key_signature_a_minor(self):
        notes_of_a_minor = [('A6', 0, 1), ('B6', 1, 2), ('C7', 2, 3), ('D7', 3, 4),
                            ('E7', 4, 5), ('F7', 5, 6), ('G7', 6, 7)]
        predicted_key_signature = infer_key_signature(notes_of_a_minor)
        self.assertEqual(predicted_key_signature, 0)

    def test_infer_key_signature_f_sharp_major(self):
        notes_of_f_sharp_major = [('F#3', 0, 1), ('G#3', 1, 2), ('A#3', 2, 3), ('B3', 3, 4),
                                  ('C#4', 4, 5), ('D#4', 5, 6), ('F4', 6, 7)]
        predicted_key_signature = infer_key_signature(notes_of_f_sharp_major)
        self.assertEqual(predicted_key_signature, 6)

    def test_infer_key_signature_f_sharp_minor(self):
        notes_of_f_sharp_minor = [('F#3', 0, 1), ('G#3', 1, 2), ('A3', 2, 3), ('B3', 3, 4),
                                  ('C#4', 4, 5), ('D4', 5, 6), ('E4', 6, 7)]
        predicted_key_signature = infer_key_signature(notes_of_f_sharp_minor)
        self.assertEqual(predicted_key_signature, 3)

    def test_quantise(self):
        quantised_value_0 = quantise(0.06249)
        self.assertAlmostEqual(quantised_value_0, 0.0)
        quantised_value_1 = quantise(0.06251)
        self.assertAlmostEqual(quantised_value_1, 0.125)


if __name__ == '__main__':
    unittest.main()
