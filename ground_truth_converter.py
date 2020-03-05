import numpy as np
from music21 import *
from audio_processor import get_spectrogram
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


def get_notes_old(score, printing=False, encoding=None):
    notes = list()
    for part in score.parts:
        for measure in part.getElementsByClass("Measure"):
            seconds_map = measure.secondsMap
            if printing:
                print(f'\n-- new measure --  {seconds_map}')
            for i in seconds_map:
                element = i.get('element')
                if type(element) == note.Note or type(element) == note.Rest:
                    encoded_pitch = get_encoded_pitch(element, encoding=encoding)
                    offset_seconds = i.get('offsetSeconds')
                    duration_seconds = i.get('durationSeconds')
                    notes.append((encoded_pitch, element.offset, element.duration.quarterLength,
                                  offset_seconds, duration_seconds))

                    if printing:
                        print(f'note: {str(encoded_pitch):<4}    '
                              f'offset: {float(element.offset):7.4f} beats ({float(offset_seconds):7.4f} s)    '
                              f'duration: {float(element.duration.quarterLength):7.4f} beats ('
                              f'{float(duration_seconds):7.4f} s)')

    return notes


def get_notes_from_xml_file(xml_file_full_path, ground_truth_duration, printing=False):
    notes = list()
    score = converter.parse(xml_file_full_path)
    last_offset_time = 0.0
    for part in score.parts:
        for i in range(len(part)):
            if type(part.secondsMap[i]['element']) is stream.Measure:
                measure = part.secondsMap[i]['element']
                measure_offset_seconds = part.secondsMap[i]['offsetSeconds']
                for item in measure.secondsMap:
                    element = item['element']
                    n = None
                    onset_time = measure_offset_seconds + item['offsetSeconds']
                    offset_time = onset_time + item['durationSeconds']
                    if last_offset_time < offset_time:
                        last_offset_time = offset_time
                    if type(element) is note.Note:
                        n = (element.nameWithOctave, onset_time, offset_time)
                    elif type(element) is note.Rest:
                        n = ('rest', onset_time, offset_time)
                    if n is not None:
                        notes.insert(0, n)

    if last_offset_time < ground_truth_duration:
        notes.insert(0, ('rest', last_offset_time, ground_truth_duration))

    notes.reverse()

    if printing:
        print(notes)

    return notes


def get_monophonic_periodogram_pitch_old(times, notes, encoding=None):

    # initialise ground-truth array with rests
    if encoding == 'one_hot':
        ground_truth = np.full(len(times), one_hot_encode_midi_pitch(REST_ENCODING), dtype=object)
    elif encoding == 'midi_pitch':
        ground_truth = np.full(len(times), REST_ENCODING, dtype=object)
    else:
        ground_truth = np.full(len(times), 'rest', dtype=object)

    # for each window, determine whether a non-rest note is sounding at that time
    # for i in range(len(times)):
    #     for n in notes:
    #         note_onset = n[3]
    #         note_offset = n[3] + n[4]
    #         if note_onset < times[i] < note_offset:
    #             ground_truth[i] = n[0]

    note_index = 0
    note_onset = notes[note_index][3]
    note_offset = notes[note_index][3] + notes[note_index][4]
    i = 0
    while i < len(times):
        # if the note currently under examination is sounding at times[i],
        if note_onset < times[i] < note_offset:
            # then add its pitch to the ground truth for time-step i, and proceed to the next time-step
            ground_truth[i] = notes[note_index][0]
            i += 1
        # otherwise, if the note currently under examination has already finished sounding,
        elif note_offset < times[i]:
            # stay at the same time-step, but move on to examine the next note in the sequence
            note_index += 1
            if note_index < len(notes):
                note_onset = notes[note_index][3]
                note_offset = notes[note_index][3] + notes[note_index][4]
            else:
                break

    return ground_truth


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
        if monophonic:
            assert len(notes_present_in_window) == 1
            ground_truth_array[i] = notes_present_in_window[0]
        else:
            ground_truth_array[i] = notes_present_in_window

    return ground_truth_array


def get_monophonic_ground_truth_old(wav, xml, encoding=None, window_size=50, printing=False):
    _, times, _ = get_spectrogram(wav, window_size=window_size)
    score = converter.parse(xml)
    notes = get_notes_old(score, encoding=encoding)
    ground_truth = get_monophonic_periodogram_pitch_old(times, notes, encoding)
    if printing:
        print(f'notes:\n{notes}\n\n'
              f'mid-point times: {times.shape} {type(times)}\n{times}\n\n'
              f'ground truth: {ground_truth.shape} {type(ground_truth)}\n{ground_truth}')
    return ground_truth


def get_monophonic_ground_truth(file_name, window_size=50, wav_path='wav_files', xml_path='xml_files', printing=False):

    _, times, _ = get_spectrogram(f'{wav_path}/{file_name}.wav', window_size=window_size)
    ground_truth_duration = len(times) * window_size / 1000.0
    notes = get_notes_from_xml_file(f'{xml_path}/{file_name}.musicxml', ground_truth_duration)
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
