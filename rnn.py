from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
from encoder import decode_label, get_number_of_unique_labels
from neural_network_trainer import save_trained_model


def set_up_sequence_to_sequence_model(encoder_input_data, decoder_input_data, decoder_target_data,
                                      latent_space_size=256, batch_size=None, epochs=500,
                                      printing=True, saving=False, save_name=None):

    number_of_unique_labels = get_number_of_unique_labels(for_rnn=True)

    # set up the encoder
    encoder_inputs = Input(shape=(encoder_input_data.shape[1], encoder_input_data.shape[2]))
    encoder = LSTM(latent_space_size, return_state=True)
    __, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # set up the decoder as an LSTM layer followed by a dense layer
    decoder_inputs = Input(shape=(None, decoder_input_data.shape[2]))
    decoder_lstm = LSTM(latent_space_size, return_sequences=True, return_state=True)
    decoder_dense = Dense(number_of_unique_labels, activation='softmax')

    # configure the decoder to predict outputs based on its direct inputs and the encoder's states
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = decoder_dense(decoder_outputs)

    # construct the model that predicts decoder outputs from encoder and decoder inputs
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    if printing:
        model.summary()

    # tensorboard = TensorBoard(log_dir=f'logs/{model_name}')
    es = EarlyStopping(patience=10, min_delta=0, restore_best_weights=True)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size, epochs=epochs, callbacks=[es], validation_split=0.2)

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


def decode_sequence(input_sequence, encoder_model, decoder_model, maximum_sequence_length):

    number_of_blanks_required = maximum_sequence_length - 1 - input_sequence.shape[1]
    bof = np.full(shape=(1, 1, input_sequence.shape[2]), fill_value=-0.5)
    bof[:, :, ::2] = 0
    eofs = np.full(shape=(1, number_of_blanks_required, input_sequence.shape[2]), fill_value=-1.0)
    eofs[:, :, ::3] = 0
    input_sequence = np.concatenate((bof, input_sequence, eofs), axis=1)
    print(f'input: {input_sequence.shape}\n{input_sequence}\n')

    # encode the input as state vectors
    states_value = encoder_model.predict(input_sequence)
    print(states_value)
    print()

    # initialise the current label to the beginning-of-file label
    current_label = 89

    # initialise the decoded sequence as an empty list
    decoded_sequence = list()

    # get the labels for end-of-file markers and blank markers
    eof_label = 90

    # while there are still predictions to be made
    while len(decoded_sequence) < maximum_sequence_length and current_label is not eof_label:

        # use the decoder to predict the next label in the sequence
        current_label_reshaped = np.full(shape=(1, 1, 1), fill_value=current_label)
        print(f'current_label_reshaped: {current_label_reshaped.shape}\n{current_label_reshaped}\n')
        next_label_probabilities, h, c = decoder_model.predict([current_label_reshaped] + states_value)
        predicted_next_label = np.argmax(next_label_probabilities)

        # stop predicting notes if the end of the file has been reached
        if predicted_next_label == eof_label:
            break
        else:
            current_label = predicted_next_label
            predicted_note = decode_label(predicted_next_label)
            decoded_sequence.insert(0, predicted_note)
            states_value = [h, c]  # update the states

    decoded_sequence = np.array(decoded_sequence)[::-1]

    return decoded_sequence


def main():
    x = np.array([[[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0.0, -1.0, -1.0, 0.0], [0.0, -1.0, -1.0, 0.0],
                   [0.0, -1.0, -1.0, 0.0]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82],
                   [0.82, 0.82, 0.82, 0.82], [0.9, 0.9, 0.9, 0.9]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0.0, -1.0, -1.0, 0.0], [0.0, -1.0, -1.0, 0.0],
                   [0.0, -1.0, -1.0, 0.0]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82],
                   [0.82, 0.82, 0.82, 0.82], [0.9, 0.9, 0.9, 0.9]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0.0, -1.0, -1.0, 0.0], [0.0, -1.0, -1.0, 0.0],
                   [0.0, -1.0, -1.0, 0.0]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82],
                   [0.82, 0.82, 0.82, 0.82], [0.9, 0.9, 0.9, 0.9]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0.0, -1.0, -1.0, 0.0], [0.0, -1.0, -1.0, 0.0],
                   [0.0, -1.0, -1.0, 0.0]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82],
                   [0.82, 0.82, 0.82, 0.82], [0.9, 0.9, 0.9, 0.9]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0.0, -1.0, -1.0, 0.0], [0.0, -1.0, -1.0, 0.0],
                   [0.0, -1.0, -1.0, 0.0]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82],
                   [0.82, 0.82, 0.82, 0.82], [0.9, 0.9, 0.9, 0.9]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0.0, -1.0, -1.0, 0.0], [0.0, -1.0, -1.0, 0.0],
                   [0.0, -1.0, -1.0, 0.0]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82],
                   [0.82, 0.82, 0.82, 0.82], [0.9, 0.9, 0.9, 0.9]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0.0, -1.0, -1.0, 0.0], [0.0, -1.0, -1.0, 0.0],
                   [0.0, -1.0, -1.0, 0.0]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82],
                   [0.82, 0.82, 0.82, 0.82], [0.9, 0.9, 0.9, 0.9]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0.0, -1.0, -1.0, 0.0], [0.0, -1.0, -1.0, 0.0],
                   [0.0, -1.0, -1.0, 0.0]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82],
                   [0.82, 0.82, 0.82, 0.82], [0.9, 0.9, 0.9, 0.9]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0.0, -1.0, -1.0, 0.0], [0.0, -1.0, -1.0, 0.0],
                   [0.0, -1.0, -1.0, 0.0]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82],
                   [0.82, 0.82, 0.82, 0.82], [0.9, 0.9, 0.9, 0.9]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0.0, -1.0, -1.0, 0.0], [0.0, -1.0, -1.0, 0.0],
                   [0.0, -1.0, -1.0, 0.0]],
                  [[0.0, -0.5, 0.0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                   [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82], [0.82, 0.82, 0.82, 0.82],
                   [0.82, 0.82, 0.82, 0.82], [0.9, 0.9, 0.9, 0.9]]
                  ])

    y = np.array([[89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
                  [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
                  [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
                  [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
                  [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
                  [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
                  [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
                  [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
                  [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
                  [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90]])

    z = np.full(shape=y.shape, fill_value=91)
    z[:, :-1] = y[:, 1:]

    # y = np.array([[89, 0, 0, 40, 40, 0, 0, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
    #               [89, 0, 0, 40, 40, 0, 0, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
    #               [89, 0, 0, 40, 40, 0, 0, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
    #               [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
    #               [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
    #               [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
    #               [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
    #               [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
    #               [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90],
    #               [89, 0, 0, 40, 40, 0, 0, 90, 90, 90], [89, 0, 0, 0, 0, 82, 82, 82, 82, 90]])
    #
    # z = np.empty(len(y), dtype=object)
    # for i in range(len(y)):
    #     y[i] = np.array(y[i])
    #     y[i] = y[i].reshape((y[i].shape[0], 1))
    #
    # for i in range(len(y)):
    #     next_labels = np.full(shape=y[i].shape, fill_value=91)
    #     next_labels[:-1, :] = y[i][1:, :]
    #     z[i] = next_labels

    print(f'x: {x.shape}\n{x}\n')
    print(f'y: {y.shape}\n{y}\n')
    print(f'z: {z.shape}\n{z}\n')

    y = y.reshape((y.shape[0], y.shape[1], 1))
    z = z.reshape((z.shape[0], z.shape[1], 1))

    encoder_model, decoder_model = set_up_sequence_to_sequence_model(x, y, z)

    test = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4], [0, 0, 0, 0],
                      [0, 0, 0, 0]]])
    ground_truth = np.array([0, 0, 40, 40, 0, 0])

    print(f'test: {test.shape}\n{test}\n')

    predicted_sequence = decode_sequence(test, encoder_model, decoder_model, 10)

    print(f'predicted sequence: {predicted_sequence.shape}\n{predicted_sequence}\n')
    print(f'ground truth: {ground_truth.shape}\n{ground_truth}\n')


if __name__ == '__main__':
    main()
