import numpy as np
from music21 import *
from audio_processor import get_spectrogram


REST_ENCODING = -1


def simple_test():
    s = converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f# a2 r4")
    print(s)


def get_score(filename):
    score = converter.parse(filename)
    return score


def get_tempo(score):
    for part in score.getElementsByClass("Part"):
        for measure in part.getElementsByClass("Measure"):
            print(measure, measure.offset)
            return measure.tempo.MetronomeMark


def print_note(note_or_rest):
    if type(note_or_rest) is note.Note:
        print(f'{note_or_rest.offset: >6}  {note_or_rest.pitch}')
    if type(note_or_rest) is note.Rest:
        print(f'{note_or_rest.offset: >6}  Rest')


def encode_midi_pitch(midi_pitch, start=21, end=108):
    pitch_range = end - start + 2
    one_hot_encoding = np.zeros(pitch_range)
    if midi_pitch == REST_ENCODING:
        one_hot_encoding[pitch_range-1] = 1
    else:
        i = pitch_range - midi_pitch + start - 2
        one_hot_encoding[i] = 1
    return one_hot_encoding


def get_notes(score, printing=False, encoding='one_hot'):
    notes = list()
    for part in score.getElementsByClass("Part"):
        for measure in part.getElementsByClass("Measure"):
            seconds_map = measure.secondsMap
            for i in seconds_map:
                # print(i)
                element = i.get('element')
                if type(element) is note.Note:
                    offset_seconds = i.get('offsetSeconds')
                    duration_seconds = i.get('durationSeconds')
                    if encoding == 'one_hot':
                        one_hot_encoding = encode_midi_pitch(element.pitch.midi)
                        notes.append((one_hot_encoding, element.offset, element.duration.quarterLength,
                                      offset_seconds, duration_seconds))
                    elif encoding == 'midi_pitch':
                        notes.append((element.pitch.midi, element.offset, element.duration.quarterLength,
                                      offset_seconds, duration_seconds))
                    else:
                        notes.append((str(element.pitch), element.offset, element.duration.quarterLength,
                                      offset_seconds, duration_seconds))
                    if printing:
                        print(f'note: {str(element.pitch): <4}  '
                              f'offset: {element.offset}  '
                              f'offsetSeconds: {offset_seconds}')
                if type(element) is note.Rest:
                    offset_seconds = i.get('offsetSeconds')
                    duration_seconds = i.get('durationSeconds')
                    if encoding == 'one_hot':
                        notes.append((encode_midi_pitch(REST_ENCODING), element.offset, element.duration.quarterLength,
                                      offset_seconds, duration_seconds))
                    elif encoding == 'midi_pitch':
                        notes.append((REST_ENCODING, element.offset, element.duration.quarterLength,
                                      offset_seconds, duration_seconds))
                    else:
                        notes.append(('Rest', element.offset, element.duration.quarterLength,
                                      offset_seconds, duration_seconds))
                    if printing:
                        print(f'note: Rest  '
                              f'offset: {element.offset}  '
                              f'offsetSeconds: {offset_seconds}')
    return notes


def get_monophonic_periodogram_note(times, notes, encoding='one_hot'):
    # print('len(times):', len(times), ' len(notes):', len(notes))
    if encoding == 'one_hot':
        ground_truth = np.empty(len(times), dtype=object)
        for i in range(len(ground_truth)):
            ground_truth[i] = encode_midi_pitch(REST_ENCODING)
    elif encoding == 'midi_pitch':
        ground_truth = np.full(len(times), REST_ENCODING, dtype=object)
    else:
        ground_truth = np.empty(len(times), dtype=object)
        for i in range(len(ground_truth)):
            ground_truth[i] = 'Rest'
    for i in range(len(times)):
        for n in notes:
            if n[3] < times[i] < n[3] + n[4]:
                ground_truth[i] = n[0]
    return ground_truth


def get_monophonic_ground_truth(wav, xml, encoding='one_hot', printing=False):
    _, _, times, _ = get_spectrogram(wav)
    score = get_score(xml)
    notes = get_notes(score, encoding=encoding)
    ground_truth = get_monophonic_periodogram_note(times, notes, encoding)
    if printing:
        print(f'{notes}\n\n{times}\n{type(times)}\n\n{ground_truth}\n{type(ground_truth)}')
    return ground_truth


def main():
    # ground_truth = get_monophonic_ground_truth("scratch-wav-files/B4.wav", "scratch-xml-files/B4.musicxml")
    # ground_truth = get_monophonic_ground_truth("scratch-wav-files/B4.wav", "scratch-xml-files/B4.musicxml",
    #                                            encoding='midi_pitch')
    ground_truth = get_monophonic_ground_truth("scratch-wav-files/B4.wav", "scratch-xml-files/B4.musicxml",
                                               encoding=None)
    print(ground_truth)
    # print(encode_midi_pitch(108))
    # print()
    # print(encode_midi_pitch(21))
    # print()
    # print(encode_midi_pitch(-1))


if __name__ == "__main__":
    main()
