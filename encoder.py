import numpy as np
from os.path import splitext
from string import digits

REST_MIDI_ENCODING = -1
BoF_MIDI_ENCODING = -2
EoF_MIDI_ENCODING = -3
REST_LABEL_ENCODING = 0
BoF_LABEL_ENCODING = 89
EoF_LABEL_ENCODING = 90

pitch_offsets = {
    'C': 0,
    'C#': 1,
    'D-': 1,
    'D': 2,
    'D#': 3,
    'E-': 3,
    'E': 4,
    'F': 5,
    'F#': 6,
    'G': 7,
    'G#': 8,
    'A-': 8,
    'A': 9,
    'A#': 10,
    'B-': 10,
    'B': 11
}

pitch_offset_names = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'E-',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'A-',
    9: 'A',
    10: 'B-',
    11: 'B'
}


key_signature_notes = {
    'C':  np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]),
    'C#': np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]),
    'D':  np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]),
    'E-': np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]),
    'E':  np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]),
    'F':  np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]),
    'F#': np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]),
    'G':  np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]),
    'A-': np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]),
    'A':  np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]),
    'B-': np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]),
    'B':  np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
}


key_signature_encodings = {
    'C':  0,
    'C#': 7,
    'D-': -5,
    'D':  2,
    'E-': -3,
    'E':  4,
    'F': -1,
    'F#': 6,
    'G-': -6,
    'G':  1,
    'A-': -4,
    'A':  3,
    'B-': -2,
    'B':  5
}


def get_relative_major(key_signature):
    pitch_offset = pitch_offsets[key_signature]
    relative_minor_pitch_offset = (pitch_offset + 3) % 12
    return pitch_offset_names[relative_minor_pitch_offset]


def get_relative_minor(key_signature):
    pitch_offset = pitch_offsets[key_signature]
    relative_minor_pitch_offset = (pitch_offset - 3) % 12
    return pitch_offset_names[relative_minor_pitch_offset]


def get_bof_artificial_periodogram(number_of_bins, printing=False):
    assert type(number_of_bins) is int and number_of_bins > 0
    artificial_periodogram = np.full(shape=number_of_bins, fill_value=-0.5)
    artificial_periodogram[::2] = 0
    if printing:
        print(artificial_periodogram)
    return artificial_periodogram


def get_eof_artificial_periodogram(number_of_bins, printing=False):
    assert type(number_of_bins) is int and number_of_bins > 0
    artificial_periodogram = np.full(shape=number_of_bins, fill_value=-1.0)
    artificial_periodogram[::3] = 0
    if printing:
        print(artificial_periodogram)
    return artificial_periodogram


def get_number_of_unique_labels(start=21, end=108, for_rnn=True, printing=False):
    number_of_unique_pitches = end - start + 1
    number_of_unique_labels = number_of_unique_pitches + 1
    if for_rnn:
        number_of_unique_labels += 3
    if printing:
        print(f'number of unique labels: {number_of_unique_labels:>3}')
    return number_of_unique_labels


def get_pitch_array(start=21, end=108):
    return np.concatenate((np.array([REST_MIDI_ENCODING]), range(start, end + 1)), 0)


def get_midi_pitch(note_name, printing=False):
    if note_name == 'rest' or note_name == 'Rest':
        return REST_MIDI_ENCODING
    elif note_name == 'BoF' or note_name == 'bof' or note_name == 'BOF':
        return BoF_MIDI_ENCODING
    elif note_name == 'EoF' or note_name == 'eof' or note_name == 'EOF':
        return EoF_MIDI_ENCODING
    if len(note_name) < 2 or 3 < len(note_name):
        print('error')
    else:

        letter = note_name[0]
        accidental = 0
        octave = int(note_name[-1])

        if len(note_name) == 3:
            if note_name[1] == '-':
                accidental = -1
            elif note_name[1] == '#':
                accidental = 1

        midi_pitch = 12 + pitch_offsets[letter] + accidental + 12 * octave

        if midi_pitch < 21 or 108 < midi_pitch:
            print(f'error: {note_name} is outside the valid range')
        else:
            if printing:
                print(f' note name = {note_name}')
                print(f'\n    letter = {letter}\naccidental = {accidental}\n    octave = {octave}')
                print(f'\nMIDI pitch = {midi_pitch}')

        return midi_pitch


def get_note_name(midi_pitch, start=21, end=108):
    if midi_pitch == REST_MIDI_ENCODING:
        return 'rest'
    elif midi_pitch == BoF_MIDI_ENCODING:
        return 'BoF'
    elif midi_pitch == EoF_MIDI_ENCODING:
        return 'EoF'
    if midi_pitch < start or end < midi_pitch:
        print('error')
    else:
        pitch_name = pitch_offset_names[midi_pitch % 12]
        octave = midi_pitch // 12 - 1
        return pitch_name + str(octave)


def decode_midi_pitch(midi_pitch, start=21, end=108):
    if midi_pitch == REST_MIDI_ENCODING:
        return 'rest'
    elif midi_pitch == BoF_MIDI_ENCODING:
        return 'BoF'
    elif midi_pitch == EoF_MIDI_ENCODING:
        return 'EoF'
    if midi_pitch < start or end < midi_pitch:
        print('error')
    else:
        pitch_name = pitch_offset_names[midi_pitch % 12]
        octave = midi_pitch // 12 - 1
        return pitch_name + str(octave)


def encode_dictionary(dictionary, current_encoding=None, desired_encoding='label'):
    for key in dictionary.keys():
        dictionary[key]['ground_truth'] = encode_ground_truth_array(dictionary[key]['ground_truth'],
                                                                    current_encoding=current_encoding,
                                                                    desired_encoding=desired_encoding)
    return dictionary


def get_one_hot_midi_pitch_index(midi_pitch, start=21, end=108, putting_rests_last=False, low_last=False):
    number_of_pitches = end - start + 1
    if midi_pitch == BoF_MIDI_ENCODING:
        i = number_of_pitches + 1
    elif midi_pitch == EoF_MIDI_ENCODING:
            i = number_of_pitches + 2
    elif midi_pitch == REST_MIDI_ENCODING:
        if putting_rests_last:
            i = number_of_pitches  # set the index to put the rest at the end of the one-hot array
        else:
            i = 0  # otherwise, set the index to put the rest at the start of the one-hot array
    else:
        if low_last:
            i = end - midi_pitch  # 'end' pitches at index 0; 'start' pitches at penultimate index
        else:
            i = midi_pitch - start  # 'end' pitches at penultimate index; 'start' pitches at index 0
        if not putting_rests_last:
            i += 1  # if rests are to go at the start, shift any other pitch along by 1

    return i


def decode_label(label):
    if label == REST_LABEL_ENCODING:
        return 'rest'
    elif label == BoF_LABEL_ENCODING:
        return 'BoF'
    elif label == EoF_LABEL_ENCODING:
        return 'EoF'
    else:
        return decode_midi_pitch(label + 20)


def midi_pitch_encode_label(label):
    if label == REST_LABEL_ENCODING:
        return REST_MIDI_ENCODING
    elif label == BoF_LABEL_ENCODING:
        return BoF_MIDI_ENCODING
    elif label == EoF_LABEL_ENCODING:
        return EoF_MIDI_ENCODING
    else:
        return label + 20


def one_hot_encode_label(label, for_rnn=False):
    midi_pitch = midi_pitch_encode_label(label)
    return one_hot_encode_midi_pitch(midi_pitch, for_rnn=for_rnn)


def label_encode_midi_pitch(midi_pitch, start=21, end=108, putting_rests_last=False, low_last=False):
    number_of_pitches = end - start + 1

    if midi_pitch == BoF_MIDI_ENCODING:
        return number_of_pitches + 1
    elif midi_pitch == EoF_MIDI_ENCODING:
        return number_of_pitches + 2
    if putting_rests_last:
        if midi_pitch == REST_MIDI_ENCODING:
            return number_of_pitches
        else:
            if low_last:
                return number_of_pitches - midi_pitch + start - 1
            else:
                return midi_pitch - start
    else:
        if midi_pitch == REST_MIDI_ENCODING:
            return 0
        else:
            if low_last:
                return number_of_pitches - midi_pitch + start
            else:
                return midi_pitch - start + 1


def one_hot_encode_midi_pitch(midi_pitch, start=21, end=108, putting_rests_last=False, low_last=False,
                              for_rnn=False):
    number_of_pitches = end - start + 1
    if for_rnn:
        one_hot_encoding = np.zeros(number_of_pitches + 3)
    else:
        one_hot_encoding = np.zeros(number_of_pitches + 1)

    i = get_one_hot_midi_pitch_index(midi_pitch, start=start, end=end,
                                     putting_rests_last=putting_rests_last, low_last=low_last)

    one_hot_encoding[i] = 1
    return one_hot_encoding


def interpret_one_hot(array, encoding='midi_pitch', start=21, end=108, putting_rests_last=False, low_last=False,
                      printing=False):
    number_of_pitches = end - start + 1
    i = np.argmax(array)
    if printing:
        print(f'The maximum value in the array occurs at index {i:>2}')
    if putting_rests_last:
        if i == number_of_pitches:
            midi_pitch = REST_MIDI_ENCODING
        else:
            if low_last:
                midi_pitch = end - i
            else:
                midi_pitch = start + i
    else:
        if i == 0:
            midi_pitch = REST_MIDI_ENCODING
        else:
            if low_last:
                midi_pitch = end - i + 1
            else:
                midi_pitch = start + i - 1

    if encoding is None:
        return get_note_name(midi_pitch)
    else:
        return midi_pitch


def encode_file_name(file_name, printing=False):

    if file_name.endswith('.wav'):
        file_name_without_extension = splitext(file_name)[0]
    else:
        file_name_without_extension = file_name

    encoded_file_name = file_name_without_extension.rstrip(digits)

    if printing:
        print(f'original file name: {file_name}')
        print(f' encoded file name: {encoded_file_name}')

    return encoded_file_name


def encode_note_name_array(ground_truth_array, encoding='midi_pitch', printing=False):
    if encoding == 'one_hot':
        encoded_ground_truth_array = np.empty(len(ground_truth_array), dtype=object)
    else:
        encoded_ground_truth_array = np.empty(len(ground_truth_array))
    for i in range(len(ground_truth_array)):
        midi_pitch = get_midi_pitch(ground_truth_array[i])
        if encoding == 'midi_pitch':
            encoded_ground_truth_array[i] = midi_pitch
        elif encoding == 'label':
            encoded_ground_truth_array[i] = label_encode_midi_pitch(midi_pitch)
        elif encoding == 'one_hot':
            encoded_ground_truth_array[i] = one_hot_encode_midi_pitch(midi_pitch)
    if printing:
        print(f'        ground_truth_array: {ground_truth_array}\n'
              f'encoded_ground_truth_array: {encoded_ground_truth_array}')
    return encoded_ground_truth_array


def encode_ground_truth_array(ground_truth_array, current_encoding=None, desired_encoding='midi_pitch',
                              for_rnn=False):
    if current_encoding is None:
        midi_pitch_encoded_array = np.vectorize(get_midi_pitch)(ground_truth_array)
        if desired_encoding is None:
            return ground_truth_array
        elif desired_encoding == 'midi_pitch':
            return midi_pitch_encoded_array
        elif desired_encoding == 'label':
            return np.vectorize(label_encode_midi_pitch)(midi_pitch_encoded_array)
        elif desired_encoding == 'one_hot':
            return np.vectorize(one_hot_encode_midi_pitch, otypes=[object])(midi_pitch_encoded_array, for_rnn=for_rnn)
    elif current_encoding == 'midi_pitch':
        if desired_encoding is None:
            return np.vectorize(get_note_name)(ground_truth_array)
        elif desired_encoding == 'midi_pitch':
            return ground_truth_array
        elif desired_encoding == 'label':
            return np.vectorize(label_encode_midi_pitch)(ground_truth_array)
        elif desired_encoding == 'one_hot':
            return np.vectorize(one_hot_encode_midi_pitch, otypes=[object])(ground_truth_array, for_rnn=for_rnn)
    elif current_encoding == 'label':
        if desired_encoding is None:
            return np.vectorize(decode_label)(ground_truth_array)
        elif desired_encoding == 'midi_pitch':
            return np.vectorize(midi_pitch_encode_label)(ground_truth_array)
        elif desired_encoding == 'label':
            return ground_truth_array
        elif desired_encoding == 'one_hot':
            return np.vectorize(one_hot_encode_label, otypes=[object])(ground_truth_array, for_rnn=for_rnn)
    elif current_encoding == 'one_hot':
        midi_pitch_encoded_array = np.vectorize(interpret_one_hot)(ground_truth_array)
        if desired_encoding is None:
            return np.vectorize(interpret_one_hot)(ground_truth_array, encoding=None)
        elif desired_encoding == 'midi_pitch':
            return midi_pitch_encoded_array
        elif desired_encoding == 'label':
            return np.vectorize(label_encode_midi_pitch)(midi_pitch_encoded_array)
        elif desired_encoding == 'one_hot':
            return ground_truth_array


def main():
    test_array = np.array(['BoF', 'rest', 'C4', 'A-1', 'F#7', 'EoF'])
    output_array = encode_ground_truth_array(test_array,
                                             current_encoding=None, desired_encoding='midi_pitch', for_rnn=True)
    print(output_array)


if __name__ == "__main__":
    main()
