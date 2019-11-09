from music21 import *
from midi2audio import FluidSynth
import random


pitch_values = {
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


def get_random_length():
    return random.randint(1, 32) / random.randint(1, 32)


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


def print_note_info(input_note):
    print(f'{input_note.octave: >2} '
          f'{input_note.name: <2} '
          # f'{note_length: >8.4f} '
          + '{: <8}'.format(str(input_note.duration.quarterLength)) + ' '
          f'{input_note.duration.type: <10} '
          f'{input_note.volume.velocity: >3}')


def generate_wav(s, file_path, sound_font):
    s_midi = s.write('midi', f'{file_path}.mid')
    fs = FluidSynth(sound_font=sound_font, sample_rate=44100)
    print(f'FluidSynth(\'{sound_font}\', sample_rate=44100)')
    print(f'fs.midi_to_audio(\'{s_midi}\', \'{file_path}.wav\')')
    s_wav = fs.midi_to_audio(f'{file_path}.mid', f'{file_path}.wav')
    return s_wav


def generate_data(saving_wav=True, saving_xml=True, printing=False):
    for octave in range(8):
        for note_pitch in range(12):
            note_length = get_random_length()
            s = get_single_note_stream(note_pitch, octave, note_length, random.randint(1, 128))
            filename = f'single_{pitch_values[note_pitch]}{octave}.xml'
            if saving_wav:
                generate_wav(s, f'scratch-wav-files/{filename}', 'FluidR3_GM.sf2')
            if saving_xml:
                s.write(fp=f'scratch-xml-files/{filename}.xml')
            if printing:
                print_note_info(s[1])


def example():
    n = note.Note('C4')
    print(f'                      n = note.Note(\'C4\')')
    print(f'                 n.name = {n.name}')
    print(f'               n.octave = {n.octave}')
    print(f'       n.nameWithOctave = {n.nameWithOctave}')
    print(f'           n.pitch.midi = {n.pitch.midi}')
    print(f'n.pitch.diatonicNoteNum = {n.pitch.diatonicNoteNum}')
    print(f'     n.pitch.pitchClass = {n.pitch.pitchClass}')
    print(f'      n.pitch.frequency = {n.pitch.frequency:.2f} Hz')


def main():
    s = get_single_note_stream(0, 4, 2, 100, 1, 5)
    # print_note_info(s[1])
    # s.show()
    # print(f'scratch-xml-files/{pitch_values[s[1].pitch.pitchClass]}{s[1].octave}.xml')
    # print(pitch_values[6])
    # example()
    generate_wav(s, 'scratch-wav-files/single_C4', 'soundfonts/FluidR3_GM.sf2')


if __name__ == "__main__":
    main()
