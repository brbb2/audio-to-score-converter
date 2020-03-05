from music21 import *
import random
import os
from midi_manager import pitch_offset_names as pitch_values
from midi_to_wav_converter import make_all_wavs


def print_note_info(input_note):
    if type(input_note) is note.Note:
        print(f'{input_note.name:<2}  '
              f'{input_note.octave:>2}  '
              + '{:<8}'.format(str(input_note.duration.quarterLength)) + ' '
              f'{input_note.duration.type:<10} '
              f'{input_note.volume.velocity:>3}')
    elif type(input_note) is note.Rest:
        print(f'rest    '
              + '{:<8}'.format(str(input_note.duration.quarterLength)) + ' '
              f'{input_note.duration.type:<10} ')


def get_random_length_old():
    return random.randint(1, 32) / random.randint(1, 32)


def get_random_length(maximum_quarter_length=4):
    return random.randint(1, 4 * maximum_quarter_length) / 4.0


def get_rest_stream():
    s = stream.Stream()
    s.append(note.Rest(quarterLength=get_random_length()))
    return s


def get_single_note_stream(note_pitch, note_octave, note_length, note_velocity,
                           start_rest_length=None, end_rest_length=None):
    s = stream.Stream()
    if start_rest_length is not None:
        s.append(note.Rest(start_rest_length))
    else:
        s.append(note.Rest(quarterLength=get_random_length()))
    n = note.Note(pitch=note_pitch, octave=note_octave)
    n.duration.quarterLength = note_length
    n.volume.velocity = note_velocity
    s.append(n)
    if end_rest_length is not None:
        s.append(note.Rest(end_rest_length))
    else:
        s.append(note.Rest(quarterLength=get_random_length()))
    return s


def generate_midi_rests(samples=10, printing=False):
    for i in range(samples):
        s = get_rest_stream()
        filename = f'rest_{i}'
        s.write('midi', f'midi_files/{filename}.mid')
        if printing:
            print(f'{filename}.mid')


def generate_midi_single_notes(samples=10, start=0, octaves=9, pitches=12, printing=False):
    for octave in range(octaves):
        for note_pitch in range(pitches):
            if octave < 1 and note_pitch < 9:
                continue  # enforce that no pitch is below A0
            if octave >= 8 and note_pitch > 0:
                break  # enforce that no pitch is above C8
            for i in range(samples):
                note_length = get_random_length()
                s = get_single_note_stream(note_pitch, octave, note_length, random.randint(1, 128))
                filename = f'single_{pitch_values[note_pitch]}{octave}_{start+i}'
                s.write('midi', f'midi_files_simple/{filename}.mid')
                if printing:
                    print(f'{filename}.mid')


def generate_wav_files_from_all_midi_files():
    make_all_wavs()


def generate_xml_files_from_all_midi_files(midi_directory='midi_files', xml_directory='xml_files'):
    for filename in os.listdir(midi_directory):
        filename, _ = os.path.splitext(filename)
        print(filename)
        s = converter.parse(f'{midi_directory}/{filename}.mid')
        s.write('musicxml', f'{xml_directory}/{filename}.musicxml')


def main():
    # print(f'scratch-xml-files/{pitch_values[s[1].pitch.pitchClass]}{s[1].octave}.xml')
    # print(pitch_values[6])
    # example()
    # generate_wav(s, 'midi_files/demo', 'soundfonts/FluidR3_GM.sf2')
    # generate_midi_single_notes(samples=10, start=0, printing=True)
    # generate_midi_rests()
    # example_note('C8')
    # get_xml_from_midi()
    # generate_midi_rests(samples=1)
    generate_xml_files_from_all_midi_files(midi_directory='midi_files_simple', xml_directory='xml_files_simple')
    pass


if __name__ == "__main__":
    main()
