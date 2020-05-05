import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from data_processor import load_rnn_data_arrays
from keras.callbacks import TensorBoard, EarlyStopping
from neural_network_trainer import save_trained_model, load_model
from encoder import decode_label, get_number_of_unique_labels, encode_ground_truth_array,\
    get_bof_artificial_periodogram, get_eof_artificial_periodogram, BoF_LABEL_ENCODING, EoF_LABEL_ENCODING


def load_rnn_model(name):
    encoder_model = load_model(f'{name}_encoder')
    decoder_model = load_model(f'{name}_decoder')
    return encoder_model, decoder_model


def set_up_sequence_to_sequence_model(encoder_inputs_train, decoder_inputs_train, decoder_outputs_train,
                                      encoder_inputs_val, decoder_inputs_val, decoder_outputs_val,
                                      latent_space_size=256, batch_size=None, epochs=500,
                                      printing=True, saving=False, save_name=None):

    number_of_unique_labels = get_number_of_unique_labels(for_rnn=True)

    # set up the encoder
    encoder_inputs = Input(shape=(encoder_inputs_train.shape[1], encoder_inputs_train.shape[2]))
    encoder = LSTM(latent_space_size, return_state=True)
    __, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # set up the decoder as an LSTM layer followed by a dense layer
    decoder_inputs = Input(shape=(None, decoder_inputs_train.shape[2]))
    decoder_lstm = LSTM(latent_space_size, return_sequences=True, return_state=True)
    decoder_dense = Dense(number_of_unique_labels, activation='softmax')

    # configure the decoder to predict outputs based on its direct inputs and the encoder's states
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = decoder_dense(decoder_outputs)

    # construct the model that predicts decoder outputs from encoder and decoder inputs
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    if printing:
        model.summary()

    tensorboard = TensorBoard(log_dir=f'logs/{save_name}')
    es = EarlyStopping(patience=10, min_delta=0, restore_best_weights=True)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit([encoder_inputs_train, decoder_inputs_train], decoder_outputs_train,
              batch_size=batch_size, epochs=epochs, callbacks=[tensorboard, es],
              validation_data=([encoder_inputs_val, decoder_inputs_val], decoder_outputs_val))

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_space_size,))
    decoder_state_input_c = Input(shape=(latent_space_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    if saving and save_name is not None:
        save_trained_model(encoder_model, f'{save_name}_encoder')
        save_trained_model(decoder_model, f'{save_name}_decoder')

    return encoder_model, decoder_model


def decode_sequence(input_sequence, encoder_model, decoder_model, maximum_sequence_length, printing=False):

    assert len(input_sequence) <= maximum_sequence_length

    # add a BoF periodogram to the front of the input and pad it with EoF periodograms at the end
    number_of_eofs_required = 1 + maximum_sequence_length - input_sequence.shape[0]
    bof = get_bof_artificial_periodogram(input_sequence.shape[1]).reshape((1, input_sequence.shape[1]))
    eof = get_eof_artificial_periodogram(input_sequence.shape[1]).reshape((1, input_sequence.shape[1]))
    eofs = np.repeat(eof, number_of_eofs_required, axis=0)

    if printing:
        print(f' bof: {bof.shape}\n{bof}\n')
        print(f'input: {input_sequence.shape}\n{input_sequence}\n')
        print(f'eofs: {eofs.shape}\n{eofs}\n')

    input_sequence = np.concatenate((bof, input_sequence, eofs), axis=0)
    input_sequence = input_sequence.reshape((1, input_sequence.shape[0], input_sequence.shape[1]))

    if printing:
        print(f'input: {input_sequence.shape}\n{input_sequence}\n')

    # encode the input as state vectors
    states_value = encoder_model.predict(input_sequence)

    # initialise the current label to the beginning-of-file label
    current_label = BoF_LABEL_ENCODING

    # initialise the decoded sequence as an empty list
    decoded_sequence = list()

    # get the labels for end-of-file markers and blank markers
    eof_label = EoF_LABEL_ENCODING

    # while there are still predictions to be made
    while len(decoded_sequence) < maximum_sequence_length and current_label is not eof_label:

        # use the decoder to predict the next label in the sequence
        current_label_reshaped = np.full(shape=(1, 1, 1), fill_value=current_label)

        if printing:
            print(f'current_label_reshaped: {current_label_reshaped.shape}\n{current_label_reshaped}\n')

        # use the current encoder state and the current label to predict the next label in the sequence
        next_label_probabilities, h, c = decoder_model.predict([current_label_reshaped] + states_value)

        # infer the most likely next label from the softmax probabilities
        predicted_next_label = np.argmax(next_label_probabilities)

        # stop predicting notes if the end of the file has been reached
        if predicted_next_label == eof_label:
            break
        else:
            current_label = predicted_next_label
            predicted_note = decode_label(predicted_next_label)  # get the note name from the label
            decoded_sequence.insert(0, predicted_note)  # add the predicted note to the sequence
            states_value = [h, c]  # update the states

    # turn the sequence from a list into an array and flip it to reverse the effect of inserting into the front
    decoded_sequence = np.array(decoded_sequence)[::-1]

    return decoded_sequence


def make_prediction(encoder_inputs_val, decoder_inputs_val, sample, model_name='rnn_label_freq_50_powers',
                    printing=True, saving=False):

    encoder_model, decoder_model = load_rnn_model(model_name)

    file_name = f'validation_sample_{sample}'

    max_length = encoder_inputs_val.shape[1] - 2
    ground_truth = decoder_inputs_val[sample][1:-1]
    ground_truth = ground_truth.reshape(ground_truth.shape[0])
    ground_truth = encode_ground_truth_array(ground_truth, current_encoding='label', desired_encoding=None)
    ground_truth_list = list()

    i = 0
    while i < len(ground_truth) and ground_truth[i] != 'EoF':
        ground_truth_list.insert(0, ground_truth[i])
        i += 1
    ground_truth = np.array(ground_truth_list)[::-1]

    predicted_sequence = decode_sequence(encoder_inputs_val[sample][1:-1], encoder_model, decoder_model, max_length)

    if printing:
        print(f'predicted sequence: {predicted_sequence.shape}\n{predicted_sequence}\n')
        print(f'ground truth: {ground_truth.shape}\n{ground_truth}\n')

    if saving:
        f = open(f'txt_files/{file_name}.txt', 'w')
        f.write('        time step:   ')
        for time_step in range(max(len(predicted_sequence), len(ground_truth))):
            f.write(f'{time_step:<5}')
        f.write('\n')
        f.write('     ground truth:   ')
        for pitch in ground_truth:
            if pitch == 'EoF':
                break
            f.write(f'{pitch:<5}')
        f.write('\n')
        f.write('model predictions:   ')
        for pitch in predicted_sequence:
            f.write(f'{pitch:<5}')
        f.close()

    return predicted_sequence


def main():
    encoder_inputs_train, decoder_inputs_train, decoder_outputs_train,\
        encoder_inputs_val, decoder_inputs_val, decoder_outputs_val = \
        load_rnn_data_arrays('rnn_label_freq_050ms_powers')
    # print_rnn_split_data(encoder_inputs_train, decoder_inputs_train, decoder_outputs_train,
    #                      encoder_inputs_val, decoder_inputs_val, decoder_outputs_val)

    # for i in range(2):
    #     file = encoder_inputs_train[i]
    #     # for periodogram in file:
    #     #     print(periodogram)
    #     print(file[60])
    #     print()

    # set_up_sequence_to_sequence_model(encoder_inputs_train, decoder_inputs_train, decoder_outputs_train,
    #                                   encoder_inputs_val, decoder_inputs_val, decoder_outputs_val,
    #                                   saving=True, save_name='rnn_label_freq_50_powers')

    make_prediction(encoder_inputs_val, decoder_inputs_val, sample=40, saving=False)


if __name__ == '__main__':
    main()
