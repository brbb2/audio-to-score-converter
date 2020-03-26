from encoder import label_encode_midi_pitch, decode_midi_pitch, decode_label, get_pitch_array
from ground_truth_converter import get_notes_from_xml_file
from data_processor import add_spectral_powers, normalise
from neural_network_trainer import load_model
from audio_processor import get_spectrogram
from music21 import *
import numpy as np
import matplotlib.pyplot as plt


def get_pitch_probabilities_for_each_window(file_name, wav_path='wav_files_simple', window_size=25,
                                            normalising=True, adding_spectral_powers=True, using_saved_maximum=True,
                                            returning_times=False, printing=False):
    model_name = f'label_freq_025ms_with_powers_remove_rests_log_k1_normalised_using_dropout_midi_model'
    # model_name = f'label_freq_{window_size:0>3}ms_with_powers_remove_rests_normalised_using_dropout'
    model = load_model(model_name)
    if wav_path is None:
        wav_file_full_path = f'{file_name}.wav'
    else:
        wav_file_full_path = f'{wav_path}/{file_name}.wav'

    # get the spectrogram data and convert it into periodogram data
    _, times, spectrogram = get_spectrogram(wav_file_full_path, window_size=window_size)
    periodograms = np.swapaxes(spectrogram, 0, 1)

    # preprocess the periodogram data
    periodograms = add_spectral_powers(periodograms)
    periodograms = periodograms.reshape(periodograms.shape[0], periodograms.shape[1], 1)
    if normalising:
        periodograms = normalise(periodograms, spectral_powers_present=adding_spectral_powers,
                                 first_order_differences_present=False, using_saved_maximum=using_saved_maximum)

    pitch_probabilities_for_each_window = model.predict([periodograms])

    if printing:
        print(spectrogram.shape)
        print(spectrogram)
        print(periodograms.shape)
        print(periodograms)

    # print(pitch_probabilities_for_each_window.shape)
    # for i in range(len(pitch_probabilities_for_each_window)):
    #     print(pitch_probabilities_for_each_window[i])

    if returning_times:
        return pitch_probabilities_for_each_window, times
    else:
        return pitch_probabilities_for_each_window


def get_pitch_probabilities_for_each_pitch(file_name, wav_path='wav_files_simple', window_size=25,
                                           normalising=True, adding_spectral_powers=True, using_saved_maximum=True,
                                           returning_times=False, printing=False):
    pitch_probabilities_for_each_window, times = \
        get_pitch_probabilities_for_each_window(file_name, wav_path=wav_path, window_size=window_size,
                                                normalising=normalising, adding_spectral_powers=adding_spectral_powers,
                                                using_saved_maximum=using_saved_maximum, returning_times=True,
                                                printing=printing)

    pitch_probabilities_for_each_pitch = np.swapaxes(pitch_probabilities_for_each_window, 0, 1)

    if returning_times:
        return pitch_probabilities_for_each_pitch, times
    else:
        return pitch_probabilities_for_each_pitch


def get_most_likely_pitch_for_each_window(file_name, wav_path='wav_files_simple', window_size=25,
                                          normalising=True, adding_spectral_powers=True, using_saved_maximum=True,
                                          returning_times=False, printing=False):
    pitch_probabilities_for_each_window, times = \
        get_pitch_probabilities_for_each_window(file_name, wav_path=wav_path, window_size=window_size,
                                                normalising=normalising, adding_spectral_powers=adding_spectral_powers,
                                                using_saved_maximum=using_saved_maximum, returning_times=True,
                                                printing=printing)
    most_likely_pitch_for_each_window = np.empty(shape=len(pitch_probabilities_for_each_window), dtype=object)
    for window in range(len(pitch_probabilities_for_each_window)):
        pitch_probabilities = pitch_probabilities_for_each_window[window]
        most_likely_pitch = decode_label(np.argmax(pitch_probabilities))
        largest_probability = np.max(pitch_probabilities)
        most_likely_pitch_tuple = (most_likely_pitch, largest_probability)
        most_likely_pitch_for_each_window[window] = most_likely_pitch_tuple
    if returning_times:
        return most_likely_pitch_for_each_window, times
    else:
        return most_likely_pitch_for_each_window


def sweep(most_likely_pitch_for_each_window, threshold=0.9, window_size=25, quantising=False, quantise_degree=0.25,
          printing=False):

    # initialise required variables
    i = 0
    current_note_name = None
    current_note_onset_time = None
    this_is_the_first_note = True
    predicted_notes = list()

    # loop through every window
    while i < len(most_likely_pitch_for_each_window):

        # inspect the predicted pitch with its given probability at the ith window
        this_note_tuple = most_likely_pitch_for_each_window[i]
        this_note_name = this_note_tuple[0]
        this_note_probability = this_note_tuple[1]

        # if a new pitch is encountered, having a probability above the user-determined threshold,
        # then interpret this to mean that a new note has been found
        if this_note_name != current_note_name and this_note_probability > threshold:
            # if the new note is the first note to be found
            if this_is_the_first_note:
                # then it has now been seen, and so new notes can no longer be the first note
                this_is_the_first_note = False
            else:
                # otherwise, add the previous current note, now that its offset time has been discovered
                current_note_offset_time = i * window_size / 1000.0
                if quantising:
                    current_note_onset_time = quantise(current_note_onset_time, quantise_degree)
                    current_note_offset_time = quantise(current_note_offset_time, quantise_degree)
                predicted_notes.insert(0, (current_note_name, current_note_onset_time, current_note_offset_time))

            # update the current-note details
            current_note_name = this_note_name
            current_note_onset_time = i * window_size / 1000.0

        i += 1

    # after the loop, get the information needed to check whether there is one last note to add
    required_duration_of_prediction = len(most_likely_pitch_for_each_window) * window_size / 1000.0
    last_found_note_name = predicted_notes[0][0]
    last_found_offset_time = predicted_notes[0][2]

    # if there is one last note to add at the end of the file
    if last_found_offset_time < required_duration_of_prediction:
        current_note_offset_time = required_duration_of_prediction
        # then add the last note found that exceeded the user-determined probability threshold
        if current_note_name != last_found_note_name:
            if quantising:
                current_note_onset_time = quantise(current_note_onset_time, quantise_degree)
                current_note_offset_time = quantise(current_note_offset_time, quantise_degree)
            predicted_notes.insert(0, (current_note_name, current_note_onset_time, current_note_offset_time))
        # if there is no sufficiently probable note, then pad the prediction to the end with a rest as a default
        else:
            predicted_notes.insert(0, ('rest', current_note_onset_time, current_note_offset_time))

    # reverse the list to counter the effect of inserting into the front
    predicted_notes.reverse()

    # notes_array = np.empty(shape=len(notes), dtype=object)
    # for i in range(len(notes)):
    #     notes_array[i] = notes[i]
    # return notes_array

    if printing:
        print(predicted_notes)

    return predicted_notes


def create_predicted_score(predicted_notes, save_name='test', bpm=120, printing=False, saving=True):
    s = stream.Stream()
    for predicted_note in predicted_notes:
        quarter_length = (predicted_note[2] - predicted_note[1]) * bpm / 60.0
        if predicted_note[0] == 'rest':
            n = note.Rest(quarterLength=quarter_length)
        else:
            n = note.Note(nameWithOctave=predicted_note[0], quarterLength=quarter_length)
        if printing:
            print(n)
        s.append(n)

    if printing:
        print(f'{s}')

    if saving:
        s.write('musicxml', f'{save_name}.musicxml')

    return s


def plot_pitch_probability_over_time(index, pitch_probabilities_for_each_window,
                                     starting_new_figure=True, showing=True):
    if starting_new_figure:
        plt.figure()
    plt.plot(pitch_probabilities_for_each_window[:, index])
    if showing:
        plt.show()


def plot_pitch_probabilities_over_time(midi_pitches, times, pitch_probabilities_for_each_pitch):
    plt.figure()
    plt.title(f'{midi_pitches}')
    plt.xlabel(f'time (seconds)')
    plt.ylabel(f'probability')
    for midi_pitch in midi_pitches:
        label = label_encode_midi_pitch(midi_pitch)
        plt.plot(times, pitch_probabilities_for_each_pitch[label], label=midi_pitch)
    plt.show()


def plot_all_pitch_probabilities_over_time(times, pitch_probabilities_for_each_pitch, file_name=None, showing=True):
    plt.figure()
    if file_name is not None:
        plt.title(f'Pitch Probabilities for\n\"{file_name}.wav\"')
    plt.xlabel(f'time (seconds)')
    plt.ylabel(f'probability')
    midi_pitches = get_pitch_array()
    for midi_pitch in midi_pitches:
        index = label_encode_midi_pitch(midi_pitch)
        note_name = decode_midi_pitch(midi_pitch)
        y = pitch_probabilities_for_each_pitch[index]
        if np.max(y) > 0.05:
            plt.plot(times, y, label=note_name)
    plt.legend()
    if showing:
        plt.show()


def quantise(value, degree=0.25):
    scale_factor = 1.0 / degree
    value *= scale_factor
    value = round(value)
    value /= scale_factor
    return value


def main():

    # file_name = f'single_C#3_0'
    # wav_path = f'wav_files_simple'
    # xml_path = f'xml_files_simple'

    file_name = f'Billie Jean Riff'
    wav_path = f'test_files/test_wavs'
    xml_path = f'test_files/test_xmls'

    wav_file_full_path = f'{wav_path}/{file_name}.wav'
    xml_file_full_path = f'{xml_path}/{file_name}.musicxml'

    threshold = 0.7
    window_size = 25

    _, times, _ = get_spectrogram(wav_file_full_path, window_size=window_size)
    ground_truth_duration = len(times) * window_size / 1000.0
    notes = get_notes_from_xml_file(xml_file_full_path, ground_truth_duration)
    pitches_and_probabilities, times = get_most_likely_pitch_for_each_window(file_name,
                                                                             wav_path=wav_path, returning_times=True)
    predicted_notes = sweep(pitches_and_probabilities, quantising=True, threshold=threshold)
    print(f'\n{file_name}')
    print(f'\nnotes picked by system: {predicted_notes}')
    print(f' notes in the XML file: {notes}')
    create_predicted_score(predicted_notes, saving=True, save_name=f'{file_name} output {threshold}')

    # graph plotting
    # pitch_probabilities_for_each_pitch, times = get_pitch_probabilities_for_each_pitch(file_name,
    #                                                                                    wav_path=wav_path,
    #                                                                                    returning_times=True,
    #                                                                                    using_saved_maximum=True)

    # plot_pitch_probability_over_time(0, pitch_probabilities_for_each_window, showing=False)
    # plot_pitch_probability_over_time(49, pitch_probabilities_for_each_window, starting_new_figure=False)
    # plot_all_pitch_probabilities_over_time(times, pitch_probabilities_for_each_pitch,
    #                                        file_name=f'{file_name}_using_saved_maximum', showing=False)
    # plot_all_pitch_probabilities_over_time(times_2, pitch_probabilities_for_each_pitch_2, file_name=file_name)
    # plot_all_pitch_probabilities_over_time(times, pitch_probabilities_for_each_pitch,
    #                                        file_name=file_name, showing=True)


if __name__ == '__main__':
    main()
