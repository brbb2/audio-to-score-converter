import numpy as np

REST_ENCODING = -1

pitch_offsets = {
    'C': 0,
    'C#': 1,
    'D': 2,
    'E-': 3,
    'E': 4,
    'F': 5,
    'G': 7,
    'A-': 8,
    'A': 9,
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


def get_pitch_array(start=21, end=108):
    return np.concatenate((np.array([REST_ENCODING]), range(start, end+1)), 0)


def get_midi_pitch(note_name, printing=False):
    if note_name == 'rest' or note_name == 'Rest':
        return REST_ENCODING
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
    if midi_pitch == REST_ENCODING:
        return 'rest'
    if midi_pitch < start or end < midi_pitch:
        print('error')
    else:
        pitch_name = pitch_offset_names[midi_pitch % 12]
        octave = midi_pitch // 12 - 1
        return pitch_name + str(octave)


def decode_midi_pitch(midi_pitch, start=21, end=108):
    if midi_pitch == REST_ENCODING:
        return 'rest'
    if midi_pitch < start or end < midi_pitch:
        print('error')
    else:
        pitch_name = pitch_offset_names[midi_pitch % 12]
        octave = midi_pitch // 12 - 1
        return pitch_name + str(octave)


def get_one_hot_midi_pitch_index(midi_pitch, start=21, end=108, putting_rests_last=False, low_last=False):
    number_of_pitches = end - start + 1

    if midi_pitch == REST_ENCODING:
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


def label_encode_midi_pitch(midi_pitch, start=21, end=108, putting_rests_last=False, low_last=False):
    number_of_pitches = end - start + 1

    if putting_rests_last:
        if midi_pitch == REST_ENCODING:
            return number_of_pitches
        else:
            if low_last:
                return number_of_pitches - midi_pitch + start - 1
            else:
                return midi_pitch - start
    else:
        if midi_pitch == REST_ENCODING:
            return 0
        else:
            if low_last:
                return number_of_pitches - midi_pitch + start
            else:
                return midi_pitch - start + 1


def one_hot_encode_midi_pitch(midi_pitch, start=21, end=108, putting_rests_last=False, low_last=False):
    number_of_pitches = end - start + 1
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
            midi_pitch = REST_ENCODING
        else:
            if low_last:
                midi_pitch = end - i
            else:
                midi_pitch = start + i
    else:
        if i == 0:
            midi_pitch = REST_ENCODING
        else:
            if low_last:
                midi_pitch = end - i + 1
            else:
                midi_pitch = start + i - 1

    if encoding is None:
        return get_note_name(midi_pitch)
    else:
        return midi_pitch


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
                              printing=False):
    if current_encoding is None:
        midi_pitch_encoded_array = np.vectorize(get_midi_pitch)(ground_truth_array)
        if desired_encoding is None:
            return ground_truth_array
        elif desired_encoding == 'midi_pitch':
            return midi_pitch_encoded_array
        elif desired_encoding == 'label':
            return np.vectorize(label_encode_midi_pitch)(midi_pitch_encoded_array)
        elif desired_encoding == 'one_hot':
            return np.vectorize(one_hot_encode_midi_pitch, otypes=[object])(midi_pitch_encoded_array)
    elif current_encoding == 'midi_pitch':
        if desired_encoding is None:
            return np.vectorize(get_note_name)(ground_truth_array)
        elif desired_encoding == 'midi_pitch':
            return ground_truth_array
        elif desired_encoding == 'label':
            return np.vectorize(label_encode_midi_pitch)(ground_truth_array)
        elif desired_encoding == 'one_hot':
            return np.vectorize(one_hot_encode_midi_pitch, otypes=[object])(ground_truth_array)
    elif current_encoding == 'label':
        if desired_encoding is None:
            pass
        elif desired_encoding == 'midi_pitch':
            pass
        elif desired_encoding == 'label':
            return ground_truth_array
        elif desired_encoding == 'one_hot':
            pass
    elif current_encoding == 'one_hot':
        midi_pitch_encoded_array = np.vectorize(interpret_one_hot, otypes=int)(ground_truth_array)
        if desired_encoding is None:
            return np.vectorize(interpret_one_hot(encoding=None), otypes=int)(ground_truth_array)
        elif desired_encoding == 'midi_pitch':
            return midi_pitch_encoded_array
        elif desired_encoding == 'label':
            return np.vectorize(label_encode_midi_pitch)(midi_pitch_encoded_array)
        elif desired_encoding == 'one_hot':
            return ground_truth_array


def main():
    test_array = np.array(['C4', 'rest', 'A-1', 'F#7'])
    a = encode_ground_truth_array(test_array, current_encoding=None, desired_encoding='label', printing=True)
    print(a)


if __name__ == "__main__":
    main()
