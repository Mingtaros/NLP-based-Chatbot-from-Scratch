from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
import numpy as np
import pickle

class SpellCheckModule:
    def __init__(self, model, typo_char_map, real_word_char_map, num_encoder_tokens, num_decoder_tokens, encoder_seq_lengths, decoder_seq_lengths):
        encoder_inputs = model.input[0]  # input_1
        encoder_lstm_1 = model.layers[1]  # lstm_encoder_1
        encoder_dropout = model.layers[2]
        encoder_lstm_2 = model.layers[4]  # lstm_encoder_2

        encoder_lstm_1_out = encoder_dropout(encoder_lstm_1(encoder_inputs))
        encoder_lstm_2_out, state_h_enc, state_c_enc = encoder_lstm_2(
            encoder_lstm_1_out)
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]  # input_2
        decoder_state_input_h = Input(shape=(150,), name="input_dec_h")
        decoder_state_input_c = Input(shape=(150,), name="input_dec_c")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[5]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[6]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

        self.typo_char_map = typo_char_map
        self.real_word_char_map = real_word_char_map

        self.num_encoder_tokens = num_encoder_tokens
        self.encoder_seq_lengths = encoder_seq_lengths
        self.num_decoder_tokens = num_decoder_tokens
        self.decoder_seq_lengths = decoder_seq_lengths

        self.reverse_input_char_index = dict(
            (i, char) for char, i in typo_char_map.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in real_word_char_map.items())

    def _fix_word(self, word_mat):
        states_value = self.encoder_model.predict(word_mat)

        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.real_word_char_map["\t"]] = 1.0

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == "\n" or len(decoded_sentence) > self.decoder_seq_lengths:
                stop_condition = True

            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            states_value = [h, c]
        return decoded_sentence

    def fix_sentence(self, sentence):
        s = [w.lower() for w in sentence.split(' ')]
        input_seq = np.zeros(
            (len(s), self.encoder_seq_lengths, self.num_encoder_tokens), dtype="float32")
        pred = []
        for w, word in enumerate(s):
            if (word.replace('-', '').replace('/', '').isdigit()):
                pred.append(word)
            else:
                unknown_char = 0
                iter = 0
                for i, char in enumerate(word):
                    if (char in self.typo_char_map):
                        input_seq[w, iter, self.typo_char_map[char]] = 1.0
                        iter += 1
                    else:
                        unknown_char += 1
                input_seq[w, iter+1:, self.typo_char_map[' ']] = 1.0

                if (float(unknown_char)/len(word) > 0.5):
                    pred.append(word)
                else:
                    pred.append(self._fix_word(input_seq[w:w+1])[:-1])
        return ' '.join(pred)


def loadSpellCheck(model_path, data_path):
    saved_data = pickle.load(open(data_path, 'rb'))
    saved_model = load_model(model_path)
    loadedSpellChecker = SpellCheckModule(
        saved_model,
        saved_data['typo_char_map'], saved_data['real_word_char_map'],
        saved_data['num_encoder_tokens'], saved_data['num_decoder_tokens'],
        saved_data['encoder_seq_lengths'], saved_data['decoder_seq_lengths']
    )
    return loadedSpellChecker
