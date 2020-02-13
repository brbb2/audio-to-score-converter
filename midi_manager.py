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


def get_midi_pitch(note_name, printing=False):
    if note_name == 'Rest':
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
        return 'Rest'
    if midi_pitch < start or end < midi_pitch:
        print('error')
    else:
        pitch__name = pitch_offset_names[midi_pitch % 12]
        octave = midi_pitch // 12 - 1
        return pitch__name + str(octave)


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


def main():
    pass


if __name__ == "__main__":
    main()
