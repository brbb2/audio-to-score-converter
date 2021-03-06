import numpy as np
from math import ceil
from os import listdir
from os.path import isfile, splitext
from music21 import converter, note, stream
from data_processor import balance_dictionary
from ground_truth_converter import get_monophonic_ground_truth
from audio_processor import get_spectrogram, get_window_parameters


def get_data_dictionary(midi_bins=False, window_size=25, saving=False, save_name=None,
                        reshaping=False):
    data = dict()

    # for each wav file in the data directory
    for filename in listdir('wav_files'):
        if isfile(f'wav_files/{filename}'):

            # remove file extension from the filename
            filename, _ = splitext(filename)

            # get the spectrogram of the audio file
            f, t, sxx = get_spectrogram(f'wav_files/{filename}.wav', using_midi_bins=midi_bins, window_size=window_size)

            # and get the ground-truth note for each periodogram in the spectrum
            ground_truth = get_monophonic_ground_truth(window_size, f'wav_files/{filename}.wav',
                                                       f'xml_files/{filename}.musicxml')

            sxx = np.swapaxes(sxx, 0, 1)

            if reshaping:
                sxx = sxx.reshape((sxx.shape[0], sxx.shape[1], 1))

            # then create an entry in the dictionary to hold these data arrays
            data[filename] = {
                'features': sxx,
                'ground_truth': ground_truth
            }

    if saving and save_name is not None:
        np.save(f'data_dictionaries/{save_name}.npy', data)

    return data


def run_test_case(test_note='A4', example=0):
    ground_truth = get_monophonic_ground_truth(f'wav_files/single_{test_note}_{example}.wav',
                                               f'xml_files/single_{test_note}_{example}.musicxml')
    print(ground_truth)


def test_missing_files():
    dictionary = get_data_dictionary()
    print(f'                    dictionary.keys(): {len(dictionary.keys())}\n{dictionary.keys()}\n')
    balanced_dictionary = balance_dictionary(dictionary)
    print(f'           balanced_dictionary.keys(): {len(balanced_dictionary.keys())}\n{balanced_dictionary.keys()}\n')
    absent_files = list()
    for key in dictionary.keys():
        if key not in balanced_dictionary.keys():
            absent_files.insert(0, key)
    absent_files.reverse()
    print(f'files absent from balanced_dictionary: {len(absent_files)}\n{absent_files}\n')
    print('\n\n')
    for absent_file in absent_files:
        print(f'{absent_file}.musicxml:')
        score = converter.parse(f'xml_files/{absent_file}.musicxml')
        print(f'{score}\n\n')


def get_single_note_onset_and_offset(ground_truth_array, window_size=50, printing=False):
    i = 0
    while i < len(ground_truth_array) and ground_truth_array[i] == 'rest':
        i += 1
    onset_time = i * window_size / 1000.0

    if printing:
        print(f'According to the ground truth,\n'
              f'the note {ground_truth_array[i]} starts in window {i} ({onset_time:6.3f} s),')

    while i < len(ground_truth_array) and ground_truth_array[i] != 'rest':
        i += 1
    offset_time = i * window_size / 1000.0

    if printing:
        print(f'and a rest occurs again in window {i} ({offset_time:6.3f} s).')

    return onset_time, offset_time


def test_getting_notes():
    notes = list()
    score = converter.parse(f'xml_files/single_A4_3.musicxml')
    file_duration = 4.0
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

    if last_offset_time < file_duration:
        notes.insert(0, ('rest', last_offset_time, file_duration))

    notes.reverse()
    print(notes)


def test_get_monophonic_ground_truth():
    get_monophonic_ground_truth('single_A4_3', wav_path='wav_files', xml_path='xml_files', printing=True)


def test_get_monophonic_ground_truth_against_known_ground_truth(window_size=25, sampling_frequency=44100):

    nperseg, noverlap = get_window_parameters(window_size)
    actual_window_size = (nperseg - noverlap) / float(sampling_frequency)
    expected_file_duration = 3.25

    ground_truth = get_monophonic_ground_truth('test_file', wav_path=None, xml_path=None)
    ground_truth_direct = get_monophonic_ground_truth('test_file_direct', wav_path=None, xml_path=None)

    expected_note_onset_time = 3.25 / 2
    expected_note_onset_window = ceil(expected_note_onset_time / actual_window_size) - 1
    expected_note_offset_time = 5.75 / 2
    expected_note_offset_window = ceil(expected_note_offset_time / actual_window_size) - 1
    actual_ground_truth_duration = len(ground_truth) * actual_window_size

    ground_truth_manual = np.full(shape=65, fill_value='rest')
    ground_truth_manual[expected_note_onset_window:expected_note_offset_window] = 'C4'

    i = 0
    while i < len(ground_truth) and ground_truth[i] == 'rest':
        i += 1
    actual_onset_window = i
    while i < len(ground_truth) and ground_truth[i] != 'rest':
        i += 1
    actual_offset_window = i

    ground_truth_onset_time = actual_onset_window * actual_window_size
    ground_truth_offset_time = actual_offset_window * actual_window_size

    assert len(ground_truth) == len(ground_truth_direct)
    for i in range(len(ground_truth)):
        assert ground_truth[i] == ground_truth_direct[i]
    for i in range(len(ground_truth_manual)):
        assert ground_truth[i] == ground_truth_manual[i]

    print()
    print(f'            desired window size: {float(window_size):6.2f} ms\n'
          f'             actual window size: {actual_window_size * 1000:6.2f} ms        '
          f'( = 2048 / 44100 = (nperseg - noverlap) / Fs )\n')

    print(f'              expected duration: {expected_file_duration:6.2f} s')
    print(f'          ground truth duration: {actual_ground_truth_duration:6.2f} s\n')

    print(f'       expected note onset time: {expected_note_onset_time:6.2f} s')
    print(f'     expected note onset window: {expected_note_onset_window:>3}')
    print(f'      expected note offset time: {expected_note_offset_time:6.2f} s')
    print(f'    expected note offset window: {expected_note_offset_window:>3}\n')

    print(f' ground truth note onset window: {actual_onset_window:>3}')
    print(f'   ground truth note onset time: {ground_truth_onset_time:6.2f} s')
    print(f'ground truth note offset window: {actual_offset_window:>3}')
    print(f'  ground truth note offset time: {ground_truth_offset_time:6.2f} s\n\n')

    print(f'ground_truth_manual: {ground_truth_manual.shape}\n{ground_truth_manual}\n')
    print(f'ground_truth: {ground_truth.shape}\n{ground_truth}\n')
    print(f'ground_truth_direct: {ground_truth_direct.shape}\n{ground_truth_direct}\n')


def main():
    test_get_monophonic_ground_truth_against_known_ground_truth()


if __name__ == "__main__":
    main()
