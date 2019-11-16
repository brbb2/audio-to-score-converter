import numpy as np
from keras.utils import normalize
from keras.models import model_from_json
from neural_network_trainer import get_data, split_data


def print_shapes(x, y, x_train, y_train, x_test, y_test):
    print(f'      x.shape: {str(x.shape): >12}          y.shape: {str(y.shape): >7}')
    print(f'x_train.shape: {str(x_train.shape): >12}    y_train.shape: {str(y_train.shape): >7}')
    print(f' x_test.shape: {str(x_test.shape): >12}     y_test.shape: {str(y_test.shape): >7}')


def print_normalisations(x):
    print('x:')
    print(x)
    print('\nnormalize(x, axis=0):')
    print(normalize(x, axis=0))
    print('\nnormalize(x, axis=1):')
    print(normalize(x, axis=1))


def print_split_data(x_train, y_train, x_test, y_test):
    print(x_train)
    print()
    print(y_train)
    print()
    print(x_test)
    print()
    print(y_test)


def print_counts_table(y, y_train, y_test):
    y_targets, y_counts = np.unique(y, return_counts=True)
    y_train_targets, y_train_counts = np.unique(y_train, return_counts=True)
    y_test_targets, y_test_counts = np.unique(y_test, return_counts=True)
    # print(y_targets, y_counts)
    # print(y_train_targets, y_train_counts)
    # print(y_test_targets, y_test_counts)
    print('value     |       y  y_train   y_test')
    for i in range(len(y_targets)):
        y_train_indices = np.where(y_train_targets == y_targets[i])[0]
        y_test_indices = np.where(y_test_targets == y_targets[i])[0]
        # print(f'{y_targets[i]} at index {y_train_indices}')
        if len(y_train_indices) > 0:
            y_train_index = y_train_indices[0]
            y_train_count = y_train_counts[y_train_index]
        else:
            y_train_count = 0
        if len(y_test_indices) > 0:
            y_test_index = y_test_indices[0]
            y_test_count = y_test_counts[y_test_index]
        else:
            y_test_count = 0
        print(f'{str(y_targets[i]): <9} | {y_counts[i]: >7}  {y_train_count: >7}  {y_test_count: >7}')


def load_scratch_model(model_name):
    json_file = open(f'scratch-models/{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f'scratch-models/{model_name}.h5')
    return loaded_model


def test_getting_data(encoding='one_hot', printing=True, deep_printing=False):
    x, y = get_data(encoding=encoding)
    x_train, y_train, x_test, y_test = split_data(x, y, printing=deep_printing)
    if printing:
        print_split_data(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test


def display_predictions(predictions, index=False):
    if index:
        return [np.argmax(predictions[x]) for x in range(len(predictions))]
    else:
        return [predictions[x][np.argmax(predictions[x])] for x in range(len(predictions))]
