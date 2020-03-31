import matplotlib.pyplot as plt
import numpy as np
from music21 import *
from string import digits
from audio_processor import get_spectrogram
from neural_network_trainer import load_model
from ground_truth_converter import get_ground_truth_notes
from data_processor import add_spectral_powers, normalise
from encoder import get_relative_major, get_relative_minor
from audio_processor import get_precise_window_duration_in_seconds
from encoder import label_encode_midi_pitch, decode_midi_pitch, decode_label, get_pitch_array
from encoder import pitch_offsets, pitch_offset_names, key_signature_notes, key_signature_encodings


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


def sweep_old(most_likely_pitch_for_each_window, threshold=0.65, window_size=25, quantising=False, quantise_degree=0.25,
              printing=False):

    window_duration = get_precise_window_duration_in_seconds(window_size)

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
            current_note_offset_time = i * window_duration
            # if the new note is the first note to be found
            if this_is_the_first_note:
                # then the first note has now been seen, and so new notes can no longer be the first note
                this_is_the_first_note = False

                # if the first note above the threshold does not start at the very beginning of the audio file
                if current_note_offset_time > 0.0:
                    # predict a rest at the beginning of the audio file by default
                    if quantising:
                        quantised_current_note_offset_time = quantise(current_note_offset_time, quantise_degree)
                        predicted_notes.insert(0, ('rest', 0.0, quantised_current_note_offset_time))
                    else:
                        predicted_notes.insert(0, ('rest', 0.0, current_note_offset_time))
            else:
                # otherwise, add the previous current note, now that its offset time has been discovered
                print(i)
                if quantising:
                    quantised_current_note_onset_time = quantise(current_note_onset_time, quantise_degree)
                    quantised_current_note_offset_time = quantise(current_note_offset_time, quantise_degree)
                    if quantised_current_note_onset_time < quantised_current_note_offset_time:
                        predicted_notes.insert(0, (current_note_name, quantised_current_note_onset_time,
                                                   quantised_current_note_offset_time))
                else:
                    predicted_notes.insert(0, (current_note_name, current_note_onset_time, current_note_offset_time))

            # update the current-note details
            current_note_name = this_note_name
            current_note_onset_time = i * window_duration

        i += 1

    # after the loop, get the information needed to check whether there is one last note to add
    required_duration_of_prediction = len(most_likely_pitch_for_each_window) * window_duration
    if len(predicted_notes) > 0:
        last_found_note_name = predicted_notes[0][0]
        last_found_offset_time = predicted_notes[0][2]
    else:
        last_found_note_name = None
        last_found_offset_time = None

    # if there is one last note to add at the end of the file
    if last_found_offset_time is None or last_found_offset_time < required_duration_of_prediction:
        if current_note_onset_time is None:
            current_note_onset_time = 0.0
        current_note_offset_time = required_duration_of_prediction
        # then add the last note found that exceeded the user-determined probability threshold
        if last_found_note_name is not None and current_note_name != last_found_note_name:
            if quantising:
                quantised_current_note_onset_time = quantise(current_note_onset_time, quantise_degree)
                quantised_current_note_offset_time = quantise(current_note_offset_time, quantise_degree)
                predicted_notes.insert(0, (current_note_name, quantised_current_note_onset_time,
                                           quantised_current_note_offset_time))
            else:
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


def sweep(pitch_probabilities_for_each_pitch, threshold, window_size=25, printing=False):

    # get the actual window duration from the approximate window size
    window_duration = get_precise_window_duration_in_seconds(window_size)

    # for each pitch, add a zero to both the start and the end of its probability array
    pitch_probabilities_for_each_pitch_modified = np.zeros(shape=(pitch_probabilities_for_each_pitch.shape[0],
                                                                  pitch_probabilities_for_each_pitch.shape[1] + 2))
    pitch_probabilities_for_each_pitch_modified[:, 1:-1] = pitch_probabilities_for_each_pitch

    # initialise and empty list for the detected notes
    notes_detected = list()
    # for each pitch, determine where and in what direction its activation function crosses the threshold
    for midi_pitch_label in range(len(pitch_probabilities_for_each_pitch)):
        note_onset = 0.0
        probability_of_pitch_over_time = pitch_probabilities_for_each_pitch_modified[midi_pitch_label]
        for i in range(pitch_probabilities_for_each_pitch.shape[1] + 1):
            # if the activation function crosses the threshold while increasing, a note onset has been found
            if probability_of_pitch_over_time[i] < threshold < probability_of_pitch_over_time[i + 1]:
                note_onset = quantise(i * window_duration)
            # if the activation function crosses the threshold while decreasing, a note offset has been found
            elif probability_of_pitch_over_time[i + 1] < threshold < probability_of_pitch_over_time[i]:
                note_offset = quantise(i * window_duration)
                note_name = decode_label(midi_pitch_label)
                if note_onset < note_offset:
                    notes_detected.insert(0, (note_name, note_onset, note_offset))

    # sort the notes by onset time
    notes_detected = sorted(notes_detected, key=lambda x: x[1])

    if printing:
        print(f'{len(notes_detected):>4} {notes_detected}')

    return notes_detected


def get_pitches_of_notes(notes):
    pitches_of_notes = list()
    for n in notes:
        pitches_of_notes.insert(0, n[0])
    pitches_of_notes.reverse()
    return pitches_of_notes


def infer_key_signature(predicted_notes, measure_quarter_length=4, bpm=120, printing=False):

    # remove irrelevant information from 'predicted_notes': namely, timing data, octaves and rests
    predicted_pitches_of_notes = get_pitches_of_notes(predicted_notes)  # remove timing data
    predicted_pitches_of_notes = list(filter(lambda x: x != 'rest', predicted_pitches_of_notes))  # remove all rests
    for i in range(len(predicted_pitches_of_notes)):  # remove the octave from the given name of each predicted note
        predicted_pitches_of_notes[i] = predicted_pitches_of_notes[i].translate({ord(k): None for k in digits})

    # for each of the 12 chromatic pitches, count how many times each pitch occurs in the predicted notes
    unique_pitches_of_notes, pitch_counts = np.unique(np.array(predicted_pitches_of_notes), return_counts=True)
    pitch_offset_counts = np.zeros(12, dtype=int)
    for i in range(len(unique_pitches_of_notes)):
        pitch_offset_counts[pitch_offsets[unique_pitches_of_notes[i]]] = pitch_counts[i]

    # record the maximum count for any pitch and find the pitches with this count
    count_of_most_common_pitch = np.max(pitch_offset_counts)
    most_common_pitches_mask = np.argwhere(pitch_offset_counts == count_of_most_common_pitch).flatten()
    most_common_pitches = list()
    for i in most_common_pitches_mask:
        most_common_pitches.insert(0, pitch_offset_names[i])
    count_of_most_common_pitch = np.max(pitch_offset_counts)
    most_common_pitches.reverse()

    # find the best major-key candidates for the predicted notes
    # by matching pitches in the list of predicted notes
    # with the pitches present in each major key
    best_major_key_signatures = list()
    best_number_of_matching_pitches = 0
    for key_signature in key_signature_notes.keys():
        key_signature_pitches = key_signature_notes[key_signature]
        number_of_matching_pitches = 0
        for i in range(12):
            if key_signature_pitches[i] > 0 and pitch_offset_counts[i] > 0:
                number_of_matching_pitches += 1
        if number_of_matching_pitches > best_number_of_matching_pitches:
            best_number_of_matching_pitches = number_of_matching_pitches
            best_major_key_signatures = list()
            best_major_key_signatures.insert(0, key_signature)
        elif number_of_matching_pitches == best_number_of_matching_pitches:
            best_major_key_signatures.insert(0, key_signature)

    # for each major key in the list of best major-key candidates, get the relative minor key
    best_minor_key_signatures = list()
    for major_key_signature in best_major_key_signatures:
        relative_minor_key = get_relative_minor(major_key_signature)
        best_minor_key_signatures.insert(0, relative_minor_key)
    best_minor_key_signatures.reverse()

    pitch_of_first_note = None
    duration_of_each_measure = measure_quarter_length * 60 / float(bpm)
    i = 0
    while i < len(predicted_pitches_of_notes):
        if predicted_notes[i][0] != 'rest' and abs((predicted_notes[i][1] / duration_of_each_measure) % 1) < 0.01:
            pitch_of_first_note = predicted_pitches_of_notes[0]
            break
        i += 1

    pitch_offset_of_first_note = pitch_offsets[pitch_of_first_note]
    count_of_first_pitch = pitch_offset_counts[pitch_offset_of_first_note]

    # TODO: improve this
    predicted_key_signature = pitch_of_first_note
    predicted_key_is_a_minor_key = pitch_of_first_note in best_minor_key_signatures

    # encode the predicted key signature to match music21's key-signature representation
    encoded_key_signature = key_signature_encodings[predicted_key_signature]

    # if the predicted key is a minor key, find the relative major key and encode that
    if predicted_key_is_a_minor_key:
        relative_major = get_relative_major(pitch_of_first_note)
        encoded_key_signature = key_signature_encodings[relative_major]

    if printing:
        print(f'    unique pitches present: {unique_pitches_of_notes}')
        print(f'              pitch counts: {pitch_offset_counts}\n')
        print(f'       most common pitches: {most_common_pitches}')
        print(f'count of most common pitch: {count_of_most_common_pitch}\n')
        print(f' first bar-beginning pitch: {pitch_of_first_note}')
        print(f'      count of first pitch: {count_of_first_pitch}\n')
        print(f' best major key signatures: {best_major_key_signatures}')
        print(f' best minor key signatures: {best_minor_key_signatures}\n')
        if predicted_key_is_a_minor_key:
            key_type = 'minor'
        else:
            key_type = 'major'
        if encoded_key_signature >= 0:
            accidentals = 'sharps'
        else:
            accidentals = 'flats'
        print(f'   predicted_key_signature: {predicted_key_signature} {key_type}')
        print(f'     encoded key signature: {encoded_key_signature} ({abs(encoded_key_signature)} {accidentals})\n')

    return encoded_key_signature


def create_predicted_score(predicted_notes, save_name='test', bpm=120, save_path='test_files/test_outputs',
                           deep_printing=False, saving=True):

    # guess the most likely key signature from the notes present in the prediction
    predicted_key_signature = infer_key_signature(predicted_notes, printing=deep_printing)

    # initialise a music21 stream of the predicted key signature
    s = stream.Stream()
    s.append(key.KeySignature(predicted_key_signature))

    # for each note, calculate its quarterLength and append it to the stream
    for predicted_note in predicted_notes:
        quarter_length = (predicted_note[2] - predicted_note[1]) * bpm / 60.0
        if predicted_note[0] == 'rest':
            n = note.Rest(quarterLength=quarter_length)
        else:
            n = note.Note(nameWithOctave=predicted_note[0], quarterLength=quarter_length)
        s.append(n)

    # if saving, save the music21 stream as a musicXML file
    if saving:
        if save_path is None:
            output_file_full_path = f'{save_name}.musicxml'
        else:
            output_file_full_path = f'{save_path}/{save_name}.musicxml'
        s.write('musicxml', output_file_full_path)

    return s


def plot_pitch_probability_over_time(index, pitch_probabilities_for_each_window,
                                     starting_new_figure=True, showing=True):
    if starting_new_figure:
        plt.figure()
    plt.plot(pitch_probabilities_for_each_window[:, index])
    plt.ylim(0.0, 1.05)
    if showing:
        plt.show()


def plot_pitch_probabilities_over_time(midi_pitches, times, pitch_probabilities_for_each_pitch):
    plt.figure()
    plt.title(f'{midi_pitches}')
    plt.xlabel(f'time (seconds)')
    plt.ylabel(f'probability')
    plt.ylim(0.0, 1.05)
    for midi_pitch in midi_pitches:
        label = label_encode_midi_pitch(midi_pitch)
        plt.plot(times, pitch_probabilities_for_each_pitch[label], label=midi_pitch)
    plt.show()


def plot_all_pitch_probabilities_over_time(times, pitch_probabilities_for_each_pitch, file_name=None,
                                           plotting_threshold=True, threshold=None, showing=True):
    plt.figure()
    if file_name is not None:
        plt.title(f'Pitch Probabilities for\n\"{file_name}.wav\"')
    plt.xlabel(f'time (seconds)')
    plt.ylabel(f'probability')
    plt.ylim(0.0, 1.05)
    midi_pitches = get_pitch_array()
    for midi_pitch in midi_pitches:
        index = label_encode_midi_pitch(midi_pitch)
        note_name = decode_midi_pitch(midi_pitch)
        y = pitch_probabilities_for_each_pitch[index]
        if np.max(y) > 0.05:
            plt.plot(times, y, label=note_name)
    if plotting_threshold and threshold is not None and 0.0 <= threshold <= 1.0:
        plt.plot([times[0], times[-1]], [threshold, threshold])
    plt.legend()
    if showing:
        plt.show()


def quantise(value, degree=0.25, quantising_time=True, bpm=120):
    if quantising_time:
        degree *= (60 / float(bpm))
    scale_factor = 1.0 / degree
    value *= scale_factor
    value = round(value)
    value /= scale_factor
    return value


def predict(file_name, threshold, wav_path='test_files/test_wavs', xml_path='test_files/test_xmls',
            window_size=25, saving=True, save_path='test_files/test_outputs', quantising=True,
            printing=False, deep_printing=False):

    # pitches_and_probabilities, times = get_most_likely_pitch_for_each_window(file_name,
    #                                                                          wav_path=wav_path, returning_times=True)
    # predicted_notes = sweep(pitches_and_probabilities, quantising=quantising, threshold=threshold)
    pitch_probabilities_for_each_pitch = get_pitch_probabilities_for_each_pitch(file_name, wav_path=wav_path,
                                                                                using_saved_maximum=True)
    predicted_notes = sweep(pitch_probabilities_for_each_pitch, threshold)

    if saving:
        create_predicted_score(predicted_notes, saving=True, save_name=f'{file_name}_output_{threshold}',
                               save_path=save_path, deep_printing=deep_printing)

    if printing:
        ground_truth_notes = get_ground_truth_notes(file_name, window_size, wav_path=wav_path, xml_path=xml_path)
        print(f'\n\"{file_name}\"')
        print(f'\nnotes picked by system: {len(predicted_notes):>4} {predicted_notes}')
        print(f' notes in the XML file: {len(ground_truth_notes):>4} {ground_truth_notes}')

    return predicted_notes


def plot_file_prediction(file_name, threshold=0.65, wav_path='test_files/test_wavs', showing=True):
    pitch_probabilities_for_each_pitch, times = get_pitch_probabilities_for_each_pitch(file_name,
                                                                                       wav_path=wav_path,
                                                                                       returning_times=True,
                                                                                       using_saved_maximum=True)
    plot_all_pitch_probabilities_over_time(times, pitch_probabilities_for_each_pitch, threshold=threshold,
                                           file_name=file_name, showing=showing)


def main():

    # file_name = f'single_G7_0'
    # wav_path = f'wav_files_simple'
    # xml_path = f'xml_files_simple'

    file_name = 'Twinkle_Twinkle_Little_Star'
    wav_path = f'test_files/test_wavs'
    xml_path = f'test_files/test_xmls'
    threshold = 0.65

    # plot_file_prediction(file_name)

    predict(file_name, threshold=threshold, saving=True, printing=True, deep_printing=True)


if __name__ == '__main__':
    main()
