import numpy as np
from keras.callbacks import LambdaCallback
from keras.layers import Input, Dense, LSTM, Masking, Concatenate, Lambda
from keras.models import Model
import random
from data import special_tokens
from model.one_hot_token_vectorizer import OneHotTokenVectorizer
from utils import stack_2d_arrays_pad_rows
import keras.backend as K


def exact_accruacy(y_true, y_pred):
    return K.min(K.cast(K.equal(y_true, y_pred), K.floatx()), axis=-1)

class RecipaModel:

    def __init__(self):
        self.token_vectorizer = None
        self.encoder_lstm = None
        self.decoder_lstm = None
        self.encoder_model = None
        self._model = None

    def _vectorize_samples(self, training_samples):
        self.token_vectorizer = OneHotTokenVectorizer()
        return self.token_vectorizer.vectorize_samples(training_samples)

    def fit(self, training_samples, test_samples=None):
        vectorized_samples = self._vectorize_samples(training_samples)
        evaluation_sample_ind = random.randrange(len(vectorized_samples))
        evaluation_sample = vectorized_samples[evaluation_sample_ind]
        vectorized_samples = vectorized_samples[:evaluation_sample_ind] + vectorized_samples[evaluation_sample_ind + 1:]
        encoder_inputs = stack_2d_arrays_pad_rows([x[0] for x in vectorized_samples])
        decoder_inputs = stack_2d_arrays_pad_rows([x[1][:-1, ...] for x in vectorized_samples])
        decoder_outputs = stack_2d_arrays_pad_rows([x[1][1:, ...] for x in vectorized_samples])

        print('Encoder input length: %d' % encoder_inputs.shape[1])
        print('Decoder input length: %d' % decoder_inputs.shape[1])

        sample = self.token_vectorizer.devectorize([evaluation_sample])[0]
        sample_y_str = ''.join(sample[1])

        def on_epoch_end(e,l):
            if test_samples:
                print('\nEpoch %d:' % e)
                self.evaluate(test_samples)

        self._build_model()
        self._model.fit(
            [encoder_inputs, decoder_inputs], [decoder_outputs],
            validation_split=0.2,
            epochs=200,
            callbacks=[
                LambdaCallback(
                    # on_batch_end=lambda b,l: print('\nBatch %d:\n ACTUAL: \'%s\'\n PREDICTED: \'%s\'\n'
                    #                                % (b,
                    #                                   sample_y_str,
                    #                                   self.predict(sample[0], as_str=True))),
                    # on_epoch_end=lambda e,l: print('\nEpoch %d:\n ACTUAL: \'%s\'\n PREDICTED: \'%s\'\n'
                    #                                % (e,
                    #                                   sample_y_str,
                    #                                   self.predict(sample[0], as_str=True))),
                    on_epoch_end=on_epoch_end,
                )
            ]
        )
        return self

    def predict(self, sample_x, limit=100, as_str=False):
        vectorized_sample_x = self.token_vectorizer.vectorize_tokens(sample_x)
        inference_input_states = self.encoder_model.predict(np.stack([vectorized_sample_x]))
        s = [special_tokens.START_TEXT]
        while s[-1] != special_tokens.END_TEXT and len(s) < limit:
            softmax_out, state_h_out, state_c_out = \
                self.decoder_model.predict([np.stack([self.token_vectorizer.vectorize_tokens([s[-1]])])] + inference_input_states)
            inference_input_states = [state_h_out, state_c_out]
            token = self.token_vectorizer.softmax_decode(softmax_out[0][0])
            s.append(token)
        if as_str:
            s = ''.join(s)
        return s

    def evaluate(self, samples, print_predictions=True):
        exact_matches = 0
        for sample in samples:
            y_pred = self.predict(sample[0], as_str=True)
            y_actual = ''.join(sample[1])
            correct = 'V' if y_pred == y_actual else 'X'
            if y_pred == y_actual:
                exact_matches += 1
            if print_predictions:
                print('ACTUAL: \'%10s\' PREDICTED: \'%10s\' %s' % (y_actual, y_pred, correct))

        exact_accruacy = exact_matches / len(samples)
        if print_predictions:
            print('EXACT_ACCURACY: %1.4f' % exact_accruacy)
        return exact_accruacy

    def _build_model(self):
        vector_length = self.token_vectorizer.get_vector_length()
        encoder_inputs = Input(shape=(None, vector_length))
        x = Masking(mask_value=0)(encoder_inputs)
        x = Dense(128, activation='sigmoid')(x)
        # self.encoder_lstm_0 = LSTM(128, return_state=False, return_sequences=True)
        # x = self.encoder_lstm_0(x)
        self.encoder_lstm_forward_1 = LSTM(128, return_state=True)
        encoder_forward_output, state_forward_h, state_forward_c = self.encoder_lstm_forward_1(x)
        self.encoder_lstm_backward_1 = LSTM(128, return_state=True)
        encoder_backward_output, state_backward_h, state_backward_c = self.encoder_lstm_forward_1(x)

        out_h_x = Concatenate()([state_forward_h, state_backward_h])
        encoder_output_h = Dense(128)(out_h_x)
        out_c_x = Concatenate()([state_forward_c, state_backward_c])
        encoder_output_c = Dense(128)(out_c_x)
        encoder_output = [encoder_output_h, encoder_output_c]
        self.encoder_model = Model(encoder_inputs, encoder_output)

        decoder_inputs = Input(shape=(None, vector_length))
        x = Masking(mask_value=0)(decoder_inputs)
        decoder_embedding = Dense(128, activation='sigmoid')
        x = decoder_embedding(x)
        self.decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
        x, _, _ = self.decoder_lstm(x, initial_state=encoder_output)
        decoder_softmax = Dense(vector_length, activation='softmax')
        decoder_outputs = decoder_softmax(x)

        inference_token_input = Input((1, vector_length))
        inference_lstm_state_h = Input((128,))
        inference_lstm_state_c = Input((128,))
        inference_input_states = [inference_lstm_state_h, inference_lstm_state_c]
        x = decoder_embedding(inference_token_input)
        x, inference_output_state_h, inference_output_state_c = self.decoder_lstm(x, initial_state=inference_input_states)
        inference_token_output = decoder_softmax(x)
        inference_output_states = [inference_output_state_h, inference_output_state_c]
        self.decoder_model = Model(
            inputs=[inference_token_input] + inference_input_states,
            outputs=[inference_token_output] + inference_output_states
        )
        self._model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])
        self._compile_model()

    def _build_model_attention(self):
        vector_length = self.token_vectorizer.get_vector_length()
        encoder_inputs = Input(shape=(None, vector_length))
        x = Masking(mask_value=0)(encoder_inputs)
        x = Dense(128, activation='sigmoid')(x)
        # self.encoder_lstm_0 = LSTM(128, return_state=False, return_sequences=True)
        # x = self.encoder_lstm_0(x)
        self.encoder_lstm_forward_1 = LSTM(128, return_state=True, return_sequences=True)
        encoder_forward_output_seq, state_forward_h, state_forward_c = self.encoder_lstm_forward_1(x)
        encoder_output = [state_forward_h, state_forward_c]
        self.encoder_model = Model(encoder_inputs, encoder_output)

        onehot_cursor = np.identity(self.max_y_token_length)
        cursor_onehot_input = Input(tensor=onehot_cursor)
        attention_mlp = Dense(self.max_x_token_length, input_dim=128 + self.max_x_token_length)

        decoder_inputs = Input(shape=(None, vector_length))
        decoder_embedding = Dense(128, activation='sigmoid')
        self.decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
        decoder_softmax = Dense(vector_length, activation='softmax')

        for i in range(self.max_y_token_length - 1):
            x = Lambda(lambda di: di[i])(decoder_inputs)
            x = decoder_embedding(x)
            x, _, _ = self.decoder_lstm(x, initial_state=encoder_output)
            decoder_outputs = decoder_softmax(x)

        inference_token_input = Input((1, vector_length))
        inference_lstm_state_h = Input((128,))
        inference_lstm_state_c = Input((128,))
        inference_input_states = [inference_lstm_state_h, inference_lstm_state_c]
        x = decoder_embedding(inference_token_input)
        x, inference_output_state_h, inference_output_state_c = self.decoder_lstm(x, initial_state=inference_input_states)
        inference_token_output = decoder_softmax(x)
        inference_output_states = [inference_output_state_h, inference_output_state_c]
        self.decoder_model = Model(
            inputs=[inference_token_input] + inference_input_states,
            outputs=[inference_token_output] + inference_output_states
        )
        self._model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])
        self._compile_model()

    def _compile_model(self):
        self._model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
            # metrics=[exact_accruacy]
        )