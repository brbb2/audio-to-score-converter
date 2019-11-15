import numpy as np
from music21 import *
from test_file import get_spectrogram


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


def get_notes(score, printing=False, midi_pitch=True):
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
                    if midi_pitch:
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
                    if midi_pitch:
                        notes.append((128, element.offset, element.duration.quarterLength,
                                      offset_seconds, duration_seconds))
                    else:
                        notes.append(('Rest', element.offset, element.duration.quarterLength,
                                      offset_seconds, duration_seconds))
                    if printing:
                        print(f'note: Rest  '
                              f'offset: {element.offset}  '
                              f'offsetSeconds: {offset_seconds}')
    return notes


def get_monophonic_periodogram_note(times, notes, midi_pitch=True):
    if midi_pitch:
        ground_truth = np.full(len(times), 128, dtype=object)
    else:
        ground_truth = np.full(len(times), 'Rest', dtype=object)
    for i in range(len(times)):
        for n in notes:
            if n[3] < times[i] < n[3] + n[4]:
                ground_truth[i] = n[0]
    return ground_truth


def get_monophonic_ground_truth(wav, xml, printing=False):
    _, _, times, _ = get_spectrogram(wav)
    score = get_score(xml)
    notes = get_notes(score)
    ground_truth = get_monophonic_periodogram_note(times, notes)
    if printing:
        print(f'{notes}\n\n{times}\n{type(times)}\n\n{ground_truth}\n{type(ground_truth)}')
    return ground_truth


def main():
    ground_truth = get_monophonic_ground_truth("scratch-wav-files/B4.wav", "scratch-xml-files/B4.musicxml")
    print(ground_truth)


if __name__ == "__main__":
    main()
