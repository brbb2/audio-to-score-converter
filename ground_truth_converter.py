import numpy as np
from music21 import *
from audio_processor import get_spectrogram


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


def get_notes_from_xml_file(xml_file_full_path, ground_truth_duration, printing=False, deep_printing=False):
    notes = list()
    score = converter.parse(xml_file_full_path)
    last_offset_time = 0.0
    for part in score.parts:
        if deep_printing:
            measure_number = 1
        for i in range(len(part)):
            if type(part.secondsMap[i]['element']) is stream.Measure:
                measure = part.secondsMap[i]['element']
                measure_offset_seconds = part.secondsMap[i]['offsetSeconds']  # get the start time of this measure
                if deep_printing:
                    print(f'MEASURE {measure_number}')
                    measure_number += 1
                for item in measure.secondsMap:
                    if deep_printing:
                        print(item)
                    element = item['element']
                    n = None
                    # measure.secondsMap gives timing offsets from the start of that measure,
                    # so add the measure's start time to get this item's onset time
                    onset_time = measure_offset_seconds + item['offsetSeconds']
                    offset_time = onset_time + item['durationSeconds']

                    if last_offset_time < offset_time:
                        last_offset_time = offset_time

                    if type(element) is note.Note:
                        n = (element.nameWithOctave, onset_time, offset_time)
                    elif type(element) is note.Rest:
                        n = ('rest', onset_time, offset_time)

                    if n is not None:
                        # if there are two rests in a row, replace the previous rest with a longer rest
                        if len(notes) > 0 and notes[0][0] == 'rest' and n[0] == 'rest':
                            notes[0] = ('rest', notes[0][1], n[2])
                        # if the current note is tied to the previous note, replace the previous note with a longer note
                        elif type(element) is note.Note and element.tie is not None and \
                                (element.tie.type == 'stop' or element.tie.type == 'continue'):
                            notes[0] = (notes[0][0], notes[0][1], n[2])
                        else:
                            notes.insert(0, n)

    if last_offset_time < ground_truth_duration:
        notes.insert(0, ('rest', last_offset_time, ground_truth_duration))

    notes = sorted(notes, key=lambda x: x[1])

    if printing:
        print(notes)

    return notes


def get_ground_truth_notes(file_name, window_size, wav_path='wav_files', xml_path='xml_files'):
    _, times, _ = get_spectrogram(f'{wav_path}/{file_name}.wav', window_size=window_size)
    ground_truth_duration = len(times) * window_size / 1000.0
    notes = get_notes_from_xml_file(f'{xml_path}/{file_name}.musicxml', ground_truth_duration)
    return notes


def get_pitches_for_each_window(times, notes, monophonic=True):

    ground_truth_array = np.empty(len(times), dtype=object)

    for i in range(len(times)):
        window_mid_point_time = times[i]
        notes_present_in_window = list()
        # find all the notes or rests that are present in the window
        for note_or_rest in notes:
            note_onset_time = note_or_rest[1]
            note_offset_time = note_or_rest[2]
            if note_onset_time < window_mid_point_time < note_offset_time:
                notes_present_in_window.insert(0, note_or_rest[0])
        notes_present_in_window.reverse()
        # print(notes_present_in_window)
        assert len(notes_present_in_window) > 0
        if monophonic:
            # assert len(notes_present_in_window) == 1
            ground_truth_array[i] = notes_present_in_window[0]
        else:
            ground_truth_array[i] = notes_present_in_window

    return ground_truth_array


def get_monophonic_ground_truth(file_name, window_size=25, wav_path='wav_files', xml_path='xml_files',
                                printing=False, deep_printing=False):

    # get the mid-point time of each window from the spectrogram of the wav file
    if wav_path is None:
        _, times, _ = get_spectrogram(f'{file_name}.wav', window_size=window_size)
    else:
        _, times, _ = get_spectrogram(f'{wav_path}/{file_name}.wav', window_size=window_size)

    ground_truth_duration = len(times) * window_size / 1000.0

    # get the notes from the musicXML file
    if xml_path is None:
        notes = get_notes_from_xml_file(f'{file_name}.musicxml', ground_truth_duration, printing=deep_printing)
    else:
        notes = get_notes_from_xml_file(f'{xml_path}/{file_name}.musicxml', ground_truth_duration,
                                        printing=deep_printing)

    # see which pitch (or rest) is present in each window
    ground_truth = get_pitches_for_each_window(times, notes)

    if printing:
        print(f'notes:\n{notes}\n\n'
              f'mid-point times: {times.shape} {type(times)}\n{times}\n\n'
              f'ground truth: {ground_truth.shape} {type(ground_truth)}\n{ground_truth}')

    return ground_truth


def main():
    s = converter.parse(f'midi_files/single_A4_7.mid')
    for part in s:
        for event in part:
            print_note(event)


if __name__ == "__main__":
    main()
