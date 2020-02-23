import numpy as np
from music21 import *
from audio_processor import get_spectrogram_scipy
from midi_manager import REST_ENCODING, one_hot_encode_midi_pitch


def print_note(event, as_tuple=True):
    if as_tuple:
        if type(event) is note.Note:
            event_name = str(event.nameWithOctave) + ','
            onset = float(event.offset) / 2
            print(f'({event_name:<5} {onset:8.3f})')
        elif type(event) is note.Rest:
            onset = float(event.offset) / 2
            print(f'(rest, {onset:8.3f})')
    else:
        if type(event) is note.Note:
            onset = float(event.offset) / 2
            print(f'{onset:8.3f}  {event.pitch}')
        if type(event) is note.Rest:
            onset = float(event.offset) / 2
            print(f'{onset:8.3f}  rest')


def get_encoded_pitch(element, encoding='one_hot'):
    encoded_pitch = None
    if type(element) is note.Note:
        if encoding == 'one_hot':
            encoded_pitch = one_hot_encode_midi_pitch(element.pitch.midi)
        elif encoding == 'midi_pitch':
            encoded_pitch = element.pitch.midi
        else:
            encoded_pitch = str(element.pitch)
    elif type(element) is note.Rest:
        if encoding == 'one_hot':
            encoded_pitch = one_hot_encode_midi_pitch(REST_ENCODING)
        elif encoding == 'midi_pitch':
            encoded_pitch = REST_ENCODING
        else:
            encoded_pitch = 'rest'
    return encoded_pitch


def get_notes(score, printing=False, encoding='one_hot'):
    notes = list()
    for part in score.getElementsByClass("Part"):
        for measure in part.getElementsByClass("Measure"):
            seconds_map = measure.secondsMap
            if printing:
                print('-- new measure --')
            for i in seconds_map:
                element = i.get('element')
                if type(element) == note.Note or type(element) == note.Rest:
                    encoded_pitch = get_encoded_pitch(element, encoding=encoding)
                    offset_seconds = i.get('offsetSeconds')
                    duration_seconds = i.get('durationSeconds')
                    notes.append((encoded_pitch, element.offset, element.duration.quarterLength,
                                  offset_seconds, duration_seconds))

                    if printing:
                        print(f'note: {str(encoded_pitch): <4}    '
                              f'offset: {float(element.offset):7.4f} beats ({float(offset_seconds):7.4f} s)    '
                              f'duration: {float(element.duration.quarterLength):7.4f} beats ('
                              f'{float(duration_seconds):7.4f} s)')

    return notes


def get_monophonic_periodogram_pitch(times, notes, encoding='midi_pitch'):

    # initialise ground-truth array with rests
    if encoding == 'one_hot':
        ground_truth = np.empty(len(times), dtype=object)
        for i in range(len(ground_truth)):
            ground_truth[i] = one_hot_encode_midi_pitch(REST_ENCODING)
    elif encoding == 'midi_pitch':
        ground_truth = np.full(len(times), REST_ENCODING, dtype=object)
    else:
        ground_truth = np.empty(len(times), dtype=object)
        for i in range(len(ground_truth)):
            ground_truth[i] = 'rest'

    # for each window, determine whether a non-rest note is sounding at that time
    for i in range(len(times)):
        for n in notes:
            note_onset = n[3]
            note_offset = n[3] + n[4]
            if note_onset < times[i] < note_offset:
                ground_truth[i] = n[0]

    return ground_truth


def get_monophonic_ground_truth(wav, xml, encoding='one_hot', nperseg=4096, noverlap=2048, printing=False):
    _, times, _ = get_spectrogram_scipy(wav, nperseg=nperseg, noverlap=noverlap)
    score = converter.parse(xml)
    notes = get_notes(score, encoding=encoding)
    ground_truth = get_monophonic_periodogram_pitch(times, notes, encoding)
    if printing:
        print(f'{notes}\n\n{times}\n{type(times)}\n\n{ground_truth}\n{type(ground_truth)}')
    return ground_truth


def main():
    s = converter.parse(f'midi_files/single_A4_7.mid')
    for part in s:
        for event in part:
            print_note(event)


if __name__ == "__main__":
    main()
