from data_pipeline.charmap_builder import CharmapBuilder
from data_pipeline.one_hot_char_encode import OneHotCharEncoder
from utils import lists_sum

class OneHotTokenVectorizer:

    def __init__(self):
        self.charmap_builder = CharmapBuilder()
        pass

    def _get_encoder(self):
        return OneHotCharEncoder(self.charmap_builder.get_charmap())

    def vectorize_samples(self, samples):
        for sample in samples:
            self.charmap_builder.add(sample[0])
            self.charmap_builder.add(sample[1])

        return [
            (self.vectorize_tokens(x), self.vectorize_tokens(y)) for (x, y) in samples
        ]

    def vectorize_tokens(self, tokens):
        return self._get_encoder().encode(tokens)

    def devectorize_tokens(self, tokens):
        return self._get_encoder().decode(tokens)


    def devectorize(self, samples):
        one_hot_char_encoder = self._get_encoder()
        return [
            (one_hot_char_encoder.decode(x), one_hot_char_encoder.decode(y)) for (x, y) in samples
        ]

    def vectorize_token(self, token):
        return self._get_encoder().encode(token)

    def devectorize_vector(self, vector):
        return self._get_encoder().decode(vector)

    def get_vector_length(self):
        return self._get_encoder().get_n_chars()

    def softmax_decode(self, softmax_vector):
        return self._get_encoder().decode_softmax(softmax_vector)