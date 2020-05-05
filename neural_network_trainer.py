import numpy as np
from neural_network_definitions import *
from keras.models import model_from_json
from encoder import interpret_one_hot, get_note_name
from keras.callbacks import TensorBoard, EarlyStopping


def print_shapes(x, y, x_train, y_train, x_test, y_test):
    print(f'      x.shape: {str(x.shape): >12}          y.shape: {str(y.shape): >7}')
    print(f'x_train.shape: {str(x_train.shape): >12}    y_train.shape: {str(y_train.shape): >7}')
    print(f' x_test.shape: {str(x_test.shape): >12}     y_test.shape: {str(y_test.shape): >7}')


def print_counts_table(y, y_train, y_test):
    y_targets, y_counts = np.unique(y, return_counts=True)
    y_train_targets, y_train_counts = np.unique(y_train, return_counts=True)
    y_test_targets, y_test_counts = np.unique(y_test, return_counts=True)
    print('value     |       y  y_train   y_test')
    for i in range(len(y_targets)):
        y_train_indices = np.where(y_train_targets == y_targets[i])[0]
        y_test_indices = np.where(y_test_targets == y_targets[i])[0]
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
    print('\n')


def print_data(x, y):
    print(f'x: {x.shape}')
    print(x)
    print(f'\ny: {y.shape}')
    print(y)


def print_split_data(x_train, y_train, x_val, y_val):
    print(f'x_train: {x_train.shape}')
    print(x_train)
    print(f'\ny_train: {y_train.shape}')
    print(y_train)
    print(f'\nx_val: {x_val.shape}')
    print(x_val)
    print(f'\ny_val: {y_val.shape}')
    print(y_val)


def load_data_arrays(save_name):
    x_train = np.load(f'data_arrays/{save_name}/x_train.npy')
    y_train = np.load(f'data_arrays/{save_name}/y_train.npy')
    x_val = np.load(f'data_arrays/{save_name}/x_val.npy')
    y_val = np.load(f'data_arrays/{save_name}/y_val.npy')
    return x_train, y_train, x_val, y_val


def get_model_definition(model_name, x_shape, dropout_rate=0.2, printing=False):
    model = None

    if model_name == 'freq':
        model = get_model_freq_dropout(x_shape, dropout_rate=dropout_rate, printing=printing)
    elif model_name == 'midi':
        model = get_model_midi_dropout(x_shape, dropout_rate=dropout_rate, printing=printing)
    elif model_name == 'rnn':
        model = get_model_midi_rnn(x_shape, printing=printing)
    elif model_name == 'freq_dense':
        model = get_model_freq_dense(x_shape, printing=printing)
    elif model_name == 'midi_dense':
        model = get_model_midi_dense(x_shape, printing=printing)
    elif model_name == 'three':
        model = get_model_three_convolutional_layers(x_shape, printing=printing)

    return model


def train_model(model, model_name, x_train, y_train, x_val, y_val, optimizer='adam', epochs=50, patience=2,
                loss='categorical_crossentropy', metrics=None, min_delta=0, saving=True):

    if metrics is None:
        metrics = ['accuracy']  # set default argument within function to avoid mutable defaults

    tensorboard = TensorBoard(log_dir=f'logs/{model_name}')
    es = EarlyStopping(patience=patience, min_delta=min_delta, restore_best_weights=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x_train, y_train, epochs=epochs, callbacks=[tensorboard, es], validation_data=(x_val, y_val))

    if saving:
        save_model_to_json(model, model_name)  # save model to JSON and save weights to HDF5

    return model


def train(model_name, save_name, x_train, y_train, x_val, y_val, optimizer='adam', epochs=500, patience=10,
          metrics=None, min_delta=0, saving=True, printing=True):

    if metrics is None:
        metrics = ['accuracy']  # set default argument within function to avoid mutable defaults

    model = get_model_definition(model_name, x_train.shape, printing=printing)
    train_model(model, save_name, x_train, y_train, x_val, y_val, optimizer=optimizer, epochs=epochs, patience=patience,
                loss='sparse_categorical_crossentropy', metrics=metrics, min_delta=min_delta, saving=saving)


def save_trained_model(model, model_name):
    model_json = model.to_json()
    with open(f'models/{model_name}.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(f'models/{model_name}.h5')


def save_model_to_json(model, model_name, scratch=False):
    model_json = model.to_json()
    if not scratch:
        with open(f'models/{model_name}.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(f'models/{model_name}.h5')
    else:
        with open(f'scratch-models/{model_name}.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(f'scratch-models/{model_name}.h5')


def load_model(model_name):
    json_file = open(f'models/{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f'models/{model_name}.h5')
    return loaded_model


def get_predictions(model, test_set):
    return model.predict([test_set])


def load_model_and_get_predictions(model_name, x):
    model = load_model(model_name)
    return model.predict([x])


def show_example_prediction(model_name, i=0, version=1, x_val=None, y_val=None, printing_in_full=False):
    if x_val is None or y_val is None:
        _, _, x_val, y_val = load_data_arrays(version)
    example = (x_val[i], y_val[i])
    model_input = example[0].reshape(1, example[0].shape[0], 1)
    example_prediction = load_model_and_get_predictions(model_name, model_input)[0]
    example_ground_truth = example[1]
    i_prediction = np.argmax(example_prediction)
    i_ground_truth = np.argmax(example_ground_truth)
    midi_pitch_prediction = interpret_one_hot(example_prediction)
    midi_pitch_ground_truth = interpret_one_hot(example_ground_truth)
    note_name_prediction = get_note_name(midi_pitch_prediction)
    note_name_ground_truth = get_note_name(midi_pitch_ground_truth)

    if printing_in_full:
        print(f'\nmodel_input: {model_input.shape}')
        print(model_input)
        print(f'\nexample_prediction: {example_prediction.shape}')
        print(example_prediction)
        print(f'\nexample_ground_truth: {example_ground_truth.shape}')
        print(example_ground_truth)

    print()
    print(f'val instance {i}')
    print(f'         i_prediction: {i_prediction:>4}            i_ground_truth: {i_ground_truth:>4}')
    print(f'midi_pitch_prediction: {midi_pitch_prediction:>4}   midi_pitch_ground_truth: {midi_pitch_ground_truth:>4}')
    print(f' note_name_prediction: {note_name_prediction:>4}    note_name_ground_truth: {note_name_ground_truth:>4}')


def show_first_n_predictions(model_name, n=25, x_val=None, y_val=None, version=1, printing_in_full=False):
    if x_val is None or y_val is None:
        _, _, x_val, y_val = load_data_arrays(version)
    assert n < len(x_val)
    for i in range(n):
        show_example_prediction(model_name, i, x_val=x_val, y_val=y_val, printing_in_full=printing_in_full)


def evaluate(model, x_test, y_test, printing=True):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    val_loss, val_acc = model.evaluate(x_test, y_test)
    if printing:
        print(f'\nval_loss = {val_loss}\nval_acc  = {val_acc}')
    return val_loss, val_acc


def load_saved_model_and_evaluate(model_name, x_test, y_test, printing=True):
    model = load_model(model_name)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    val_loss, val_acc = model.evaluate(x_test, y_test)
    if printing:
        print(f'\nval_loss = {val_loss}\nval_acc  = {val_acc}')
    return val_loss, val_acc


def load_dictionary(dictionary_name):
    return np.load(f'data_dictionaries/{dictionary_name}.npy').item()


def get_maximum(dictionary=None, dictionary_name='data_basic', printing=False):
    if dictionary is None:
        dictionary = load_dictionary(dictionary_name)
    maximum = 0
    for key in dictionary.keys():
        features = dictionary[key]['features']
        features_maximum = np.amax(features)
        if printing:
            print(f'{features_maximum} > {maximum}')
        if features_maximum > maximum:
            maximum = features_maximum
    if printing:
        print(f'\nmaximum: {maximum}')
    return maximum


def main():
    x_train, y_train, x_val, y_val = load_data_arrays('label_midi_025ms_with_powers_remove_rests_normalised')
    print_split_data(x_train, y_train, x_val, y_val)

    train('midi', 'test',
          x_train, y_train, x_val, y_val, printing=True)


if __name__ == "__main__":
    main()
