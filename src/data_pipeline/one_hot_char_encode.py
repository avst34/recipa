import numpy as np

class OneHotCharEncoder:

    def __init__(self, charmap):
        self.charmap = charmap
        self.reverse_charmap = {}
        for (t,c) in self.charmap.items():
            self.reverse_charmap[c] = t

    def get_n_chars(self):
        return len(self.charmap)

    def encode(self, s):
        m = np.zeros((len(s), self.get_n_chars()))
        charcodes = np.array([self.charmap[c] for c in s])
        m[np.arange(len(s)), charcodes] = 1
        return m

    def decode(self, arr):
        return [self.reverse_charmap[code] for code in np.dot(arr, np.arange(self.get_n_chars()))]

    def decode_softmax(self, softmax_array):
        return self.reverse_charmap[np.argmax(softmax_array, axis=0)]

