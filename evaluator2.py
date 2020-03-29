from predictor2 import *
from os import listdir
from os.path import isfile, splitext
from ground_truth_converter import get_ground_truth_notes


def get_precision(ground_truth_notes, predicted_notes, printing=False):
    number_of_predicted_notes_correct = 0

    for predicted_note in predicted_notes:
        predicted_note_name = predicted_note[0]
        predicted_note_onset_time = predicted_note[1]
        for ground_truth_note in ground_truth_notes:
            ground_truth_note_name = ground_truth_note[0]
            ground_truth_note_onset_time = ground_truth_note[1]
            if ground_truth_note_name == predicted_note_name \
                    and predicted_note_onset_time - 0.05 < ground_truth_note_onset_time \
                    < predicted_note_onset_time + 0.05:
                number_of_predicted_notes_correct += 1
                break

    precision = number_of_predicted_notes_correct / float(len(predicted_notes))

    if printing:
        print(f'\nnotes picked by system: {predicted_notes}')
        print(f' notes in the XML file: {ground_truth_notes}')
        print(f'\n             precision: {precision:6.4f}')

    return precision


def get_recall(ground_truth_notes, predicted_notes, printing=False):
    number_of_correctly_found_notes = 0

    for ground_truth_note in ground_truth_notes:
        ground_truth_note_name = ground_truth_note[0]
        ground_truth_note_onset_time = ground_truth_note[1]
        for predicted_note in predicted_notes:
            predicted_note_name = predicted_note[0]
            predicted_note_onset_time = predicted_note[1]
            if predicted_note_name == ground_truth_note_name \
                    and ground_truth_note_onset_time - 0.05 < predicted_note_onset_time \
                    < ground_truth_note_onset_time + 0.05:
                number_of_correctly_found_notes += 1
                break

    recall = number_of_correctly_found_notes / float(len(ground_truth_notes))

    if printing:
        print(f'\nnotes picked by system: {predicted_notes}')
        print(f' notes in the XML file: {ground_truth_notes}')
        print(f'\n                recall: {recall:6.4f}')

    return recall


def get_f_score(ground_truth_notes, predicted_notes, precision=None, recall=None, beta=1, printing=False):
    if precision is None:
        precision = get_precision(ground_truth_notes, predicted_notes)
    if recall is None:
        recall = get_recall(ground_truth_notes, predicted_notes)

    if precision == 0.0 and recall == 0.0:
        f_score = 0.0
    else:
        f_score = (1 + beta) * ((precision * recall) / (beta * precision + recall))

    if printing:
        print(f'\nnotes picked by system: {predicted_notes}')
        print(f' notes in the XML file: {ground_truth_notes}')
        print(f'\n             precision: {precision:6.4f}')
        print(f'\n                recall: {recall:6.4f}')
        print(f'\n               F-score: {f_score:6.4f}')

    return f_score


def evaluate_prediction(file_name, threshold, wav_path='test_files/test_wavs', xml_path='test_files/test_xmls',
                        window_size=25, printing=False):

    ground_truth_notes = get_ground_truth_notes(file_name, wav_path=wav_path, xml_path=xml_path,
                                                window_size=window_size)

    # get the most likely pitch for each window (along with its given probability)
    pitches_and_probabilities, times = get_most_likely_pitch_for_each_window(file_name, wav_path=wav_path,
                                                                             returning_times=True)

    # use the pitches of all the windows to predict the notes
    predicted_notes = sweep(pitches_and_probabilities, quantising=True, threshold=threshold)

    if printing:
        print(f'{file_name}')
    f_score = get_f_score(ground_truth_notes, predicted_notes, printing=printing)

    return f_score


def evaluate_all(test_file_names, threshold, wav_path='test_files/test_wavs', xml_path='test_files/test_xmls',
                 window_size=25, printing=False, deep_printing=False):
    accuracies = np.zeros(len(test_file_names))
    for i in range(len(test_file_names)):
        test_file_name = test_file_names[i]
        accuracy = evaluate_prediction(test_file_name, threshold=threshold, window_size=window_size,
                                       printing=deep_printing)
        accuracies[i] = accuracy

    average_accuracy = np.mean(accuracies)
    if printing:
        print(f'      accuracies: {accuracies}')
        print(f'average accuracy: {average_accuracy:6.4f}')
    return average_accuracy


def get_test_file_names(wav_path='test_files/test_wavs'):
    test_file_names = list()
    for file_name in listdir(wav_path):
        if isfile(f'{wav_path}/{file_name}'):
            file_name, _ = splitext(file_name)
            test_file_names.insert(0, file_name)
    test_file_names.reverse()
    return test_file_names


def brute_force_search_for_optimal_threshold_value(increment=0.01, start_threshold=0.0, printing=False):
    test_files = get_test_file_names()
    optimal_threshold_value_so_far = 0.0
    best_accuracy_so_far = 0.0
    threshold = start_threshold
    if printing:
        print(f'\n             threshold   F-score\n             -------------------')
    while threshold <= 1.0:
        accuracy_for_this_threshold_value = evaluate_all(test_files, threshold=threshold)
        if printing:
            print(f'             {threshold:10.8f}: {accuracy_for_this_threshold_value:6.4f}')
        if accuracy_for_this_threshold_value > best_accuracy_so_far:
            optimal_threshold_value_so_far = threshold
            best_accuracy_so_far = accuracy_for_this_threshold_value
        threshold += increment
    if printing:
        print(f'\noptimal threshold value: {optimal_threshold_value_so_far}')
        print(f'          best accuracy: {best_accuracy_so_far:6.4f}')
    return optimal_threshold_value_so_far


def main():
    # evaluate_prediction('In the Bleak Midwinter', threshold=0.7, printing=True)
    # test_file_names = get_test_file_names()
    # evaluate_all(test_file_names, threshold=0.7, printing=True)
    brute_force_search_for_optimal_threshold_value(start_threshold=0.95, increment=0.05, printing=True)


if __name__ == '__main__':
    main()
