from predictor import *
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

    if len(predicted_notes) > 0:
        precision = number_of_predicted_notes_correct / float(len(predicted_notes))
    else:
        precision = 0.0

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

    if len(predicted_notes) > 0:
        recall = number_of_correctly_found_notes / float(len(ground_truth_notes))
    else:
        recall = 0.0

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


def predict_and_evaluate(file_name, threshold, wav_path='test_files/test_wavs', xml_path='test_files/test_xmls',
                         window_size=25, also_returning_precision_and_recall=False, printing=False):

    ground_truth_notes = get_ground_truth_notes(file_name, wav_path=wav_path, xml_path=xml_path,
                                                window_size=window_size)

    predicted_notes = predict(file_name, threshold=threshold, wav_path=wav_path, saving=False)

    if also_returning_precision_and_recall:
        precision = get_precision(ground_truth_notes, predicted_notes, printing=printing)
        recall = get_recall(ground_truth_notes, predicted_notes, printing=printing)
    else:
        precision = None
        recall = None

    f_score = get_f_score(ground_truth_notes, predicted_notes, printing=printing)

    pitches_of_ground_truth_notes = get_pitches_of_notes(ground_truth_notes)
    pitches_of_predicted_notes = get_pitches_of_notes(predicted_notes)

    if printing:
        print(f'\n{file_name}')
        print(f'\nnotes picked by system: {len(predicted_notes):>4} {predicted_notes}')
        print(f' notes in the XML file: {len(ground_truth_notes):>4} {ground_truth_notes}')
        print(f'\nnotes picked by system: {len(pitches_of_predicted_notes):>4} {pitches_of_predicted_notes}')
        print(f' notes in the XML file: {len(pitches_of_ground_truth_notes):>4} {pitches_of_ground_truth_notes}\n')
        if also_returning_precision_and_recall:
            print(f'             precision: {precision:6.4f}')
            print(f'                recall: {recall:6.4f}')
        print(f'               F-score: {f_score:6.4f}')
    if also_returning_precision_and_recall:
        return f_score, precision, recall
    else:
        return f_score


def evaluate_all(test_file_names, threshold, wav_path='test_files/test_wavs', xml_path='test_files/test_xmls',
                 window_size=25, tracking_precision_and_recall=False, printing=False, deep_printing=False):
    accuracies = np.zeros(len(test_file_names))
    if tracking_precision_and_recall:
        precisions = np.zeros(len(test_file_names))
        recalls = np.zeros(len(test_file_names))
    else:
        precisions = None
        recalls = None
    for i in range(len(test_file_names)):
        test_file_name = test_file_names[i]
        if tracking_precision_and_recall:
            accuracy, precision, recall = predict_and_evaluate(test_file_name, threshold=threshold,
                                                               window_size=window_size,
                                                               wav_path=wav_path, xml_path=xml_path,
                                                               also_returning_precision_and_recall=True,
                                                               printing=deep_printing)
            precisions[i] = precision
            recalls[i] = recall
        else:
            accuracy = predict_and_evaluate(test_file_name, threshold=threshold, window_size=window_size,
                                            wav_path=wav_path, xml_path=xml_path, printing=deep_printing)
        accuracies[i] = accuracy

    average_accuracy = np.mean(accuracies)
    if tracking_precision_and_recall:
        average_precision = np.mean(precisions)
        average_recall = np.mean(recalls)
    else:
        average_precision = None
        average_recall = None
    if printing:
        if tracking_precision_and_recall:
            print(f'       precisions: {precisions}')
            print(f'average precision: {average_precision:6.4f}')
            print(f'          recalls: {recalls}')
            print(f'   average recall: {average_recall:6.4f}')
        print(f'       accuracies: {accuracies}')
        print(f' average accuracy: {average_accuracy:6.4f}')
    if tracking_precision_and_recall:
        return average_accuracy, average_precision, average_recall
    else:
        return average_accuracy


def get_test_file_names(wav_path='test_files/test_wavs'):
    test_file_names = list()
    for file_name in listdir(wav_path):
        if isfile(f'{wav_path}/{file_name}'):
            file_name, _ = splitext(file_name)
            test_file_names.insert(0, file_name)
    test_file_names.reverse()
    return test_file_names


def brute_force_search_for_optimal_threshold_value(increment=0.01, start_threshold=0.0,
                                                   wav_path='test_files/test_wavs', xml_path='test_files/test_xmls',
                                                   tracking_precision_and_recall=False, printing=False):
    test_files = get_test_file_names()
    optimal_threshold_value_so_far = 0.0
    best_accuracy_so_far = 0.0
    corresponding_precision = 0.0
    corresponding_recall = 0.0
    threshold = start_threshold
    if printing:
        if tracking_precision_and_recall:
            print(f'\n             threshold    F-score  precision  recall'
                  f'\n             ---------------------------------------')
        else:
            print(f'\n             threshold    F-score'
                  f'\n             --------------------')
    while threshold <= 1.0:
        if tracking_precision_and_recall:
            accuracy_for_this_threshold_value, precision_for_this_threshold_value, recall_for_this_threshold_value = \
                evaluate_all(test_files, threshold=threshold, tracking_precision_and_recall=True,
                             wav_path=wav_path, xml_path=xml_path)
        else:
            precision_for_this_threshold_value = None
            recall_for_this_threshold_value = None
            accuracy_for_this_threshold_value = evaluate_all(test_files, threshold=threshold, wav_path=wav_path)
        if printing:
            if tracking_precision_and_recall:
                print(f'             {threshold:10.8f}:  {accuracy_for_this_threshold_value:6.4f}    '
                      f'{precision_for_this_threshold_value:6.4f}    {recall_for_this_threshold_value:6.4f}')
            else:
                print(f'             {threshold:10.8f}:  {accuracy_for_this_threshold_value:6.4f}')
        if accuracy_for_this_threshold_value > best_accuracy_so_far:
            optimal_threshold_value_so_far = threshold
            best_accuracy_so_far = accuracy_for_this_threshold_value
            if tracking_precision_and_recall:
                corresponding_precision = precision_for_this_threshold_value
                corresponding_recall = recall_for_this_threshold_value
        threshold += increment
    if printing:
        print(f'\noptimal threshold value: {optimal_threshold_value_so_far}')
        print(f'          best accuracy: {best_accuracy_so_far:6.4f}')
        if tracking_precision_and_recall:
            print(f'corresponding precision: {corresponding_precision:6.4f}')
            print(f'   corresponding recall: {corresponding_recall:6.4f}')
    return optimal_threshold_value_so_far


def main():
    test_file_names = get_test_file_names()
    evaluate_all(test_file_names, threshold=0.65, wav_path='test_files/test_wavs', printing=True)


if __name__ == '__main__':
    main()
