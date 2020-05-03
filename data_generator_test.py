from data_generator import *
from ground_truth_converter import get_notes_from_xml_file, get_monophonic_ground_truth


def print_xml_file(xml_file):
    print(f'\n{xml_file}:\n')
    f = open(xml_file, 'r')
    lines = f.readlines()
    for line in lines:
        print(line, end='')


def test_note_length(note_length, note_pitch='C', note_octave=4):

    # create a music21 stream
    print(f'\nmusic21 stream:\n')
    s = stream.Stream()
    s.append(note.Rest(quarterLength=1))
    s.append(note.Note(pitch=note_pitch, octave=note_octave, quarterLength=note_length))
    s.append(note.Rest(quarterLength=1))
    for n in s:
        encoded_pitch = None
        if type(n) == note.Note:
            encoded_pitch = str(n.pitch)
        elif type(n) == note.Rest:
            encoded_pitch = 'rest'
        print(f'note: {encoded_pitch:<4}    '
              f'offset: {float(n.offset):7.4f} beats ({float(-1.0):7.4f} s)    '
              f'duration: {float(n.duration.quarterLength):7.4f} beats '
              f'({float(n.duration.quarterLength / 2):7.4f} s)')
    print()

    # create a MIDI file from the music21 stream
    print(f'\nMIDI file:\n')
    s.write('midi', f'scratch_files/zero_duration_test.mid')
    s = converter.parse(f'scratch_files/zero_duration_test.mid')
    for part in s.getElementsByClass("Part"):
        for n in part:
            encoded_pitch = None
            if type(n) == note.Note:
                encoded_pitch = str(n.pitch)
            elif type(n) == note.Rest:
                encoded_pitch = 'rest'
            print(f'note: {encoded_pitch:<4}    '
                  f'offset: {float(n.offset):7.4f} beats ({float(-1.0):7.4f} s)    '
                  f'duration: {float(n.duration.quarterLength):7.4f} beats '
                  f'({float(n.duration.quarterLength / 2):7.4f} s)')
    print()

    # create a musicXML file from the MIDI file
    print(f'\nmusicXML file:')
    s = converter.parse(f'scratch_files/zero_duration_test.mid')
    s.write('musicxml', f'scratch_files/zero_duration_test.musicxml')
    s = converter.parse(f'scratch_files/zero_duration_test.musicxml')
    get_notes_from_xml_file(s, printing=True)

    print()
    print_xml_file('scratch_files/zero_duration_test.musicxml')


def test_get_random_length(number_of_trials=100000, old=False):
    number_of_very_long_notes = 0
    number_of_very_short_notes = 0
    number_of_zero_notes = 0
    for trial in range(1, number_of_trials+1):
        if not old:
            random_length = get_random_length()
        else:
            random_length = get_random_length_old()
        if trial % 5000 == 0:
            print(f'{trial:>8}: {random_length}')
        if -0.001 < random_length < 0.001:
            number_of_zero_notes += 1
        elif random_length < 0.0625:
            number_of_very_short_notes += 1
        elif random_length > 4:
            number_of_very_long_notes += 1

    print(f'\n\n     proportion of notes longer than a whole note: '
          f'{number_of_very_long_notes / float(number_of_trials)}\n'
          f'proportion of notes shorter than a sixteenth note: '
          f'{number_of_very_short_notes / float(number_of_trials)}\n'
          f'      proportion of notes with a duration of zero: '
          f'{number_of_zero_notes / float(number_of_trials)}')


def example():
    example_note('C4')


def test_writing_and_reading_midi_files(note_pitch=0, octave=4, adding_final_note=False, showing=False):
    s = stream.Stream()
    s.append(note.Rest(quarterLength=10))
    n = note.Note(pitch=note_pitch, octave=octave)
    n.duration.quarterLength = 6
    n.volume.velocity = 60
    s.append(n)
    s.append(note.Rest(quarterLength=4))
    if adding_final_note:
        n2 = note.Note(pitch=note_pitch, octave=octave)
        n2.duration.quarterLength = 0.125
        n2.volume.velocity = 1
        s.append(n2)

    print('Stream before saving to MIDI file:')
    for i in s:
        if type(i) is note.Note:
            p = str(i.nameWithOctave) + ','
            onset = float(i.offset) / 2
            print(f'({p:<5} {onset:8.3f})')
        elif type(i) is note.Rest:
            onset = float(i.offset) / 2
            print(f'({i.name}, {onset:8.3f})')
        else:
            print(f'Unwanted object found: {i}')

    s.write('midi', f'scratch_files/test_concrete.mid')
    s.write('xml', f'scratch_files/test_concrete.musicxml')
    if showing:
        s.show('midi')
    m = converter.parse(f'scratch_files/test_concrete.mid')
    print('\nStream after reading back in from MIDI file:')
    for j in m:
        for i in j:
            if type(i) is note.Note:
                p = str(i.nameWithOctave) + ','
                onset = float(i.offset) / 2
                print(f'({p:<5} {onset:8.3f})')
            elif type(i) is note.Rest:
                onset = float(i.offset) / 2
                print(f'({i.name}, {onset:8.3f})')
            else:
                print(f'Unwanted object found: {i}')
    return s


def example_note(note_name):
    n = note.Note(note_name)
    print(f'                      n = note.Note(\'{note_name}\')')
    print(f'                 n.name = {n.name}')
    print(f'               n.octave = {n.octave}')
    print(f'       n.nameWithOctave = {n.nameWithOctave}')
    print(f'           n.pitch.midi = {n.pitch.midi}')
    print(f'n.pitch.diatonicNoteNum = {n.pitch.diatonicNoteNum}')
    print(f'     n.pitch.pitchClass = {n.pitch.pitchClass}')
    print(f'      n.pitch.frequency = {n.pitch.frequency:.2f} Hz')


def print_notes_from_xml_file(xml_file, xml_directory='xml_files'):
    s = converter.parse(f'{xml_directory}/{xml_file}.musicxml')
    get_notes_from_xml_file(s, printing=True)


def print_proportion_of_valid_xml_files():
    successes = 0
    file_names = os.listdir('gen_xmls')
    for file_name in file_names:
        file_name = file_name[:-9]
        print(file_name, end=' ')
        try:
            get_monophonic_ground_truth(file_name=file_name, window_size=25, wav_path='gen_wavs', xml_path='gen_xmls',
                                        printing=False, deep_printing=False)
            print('success')
            successes += 1
        except AssertionError:
            print('failure')
            continue
    print(f'\n{successes} / {len(file_names)}')


def main():
    print_notes_from_xml_file('single_A4_2', xml_directory='xml_files_simple')


if __name__ == '__main__':
    main()
