class CharmapBuilder:

    def __init__(self):
        self.charmap = {}
        self.reverse_charmap = {}

    def add(self, s):
        for c in s:
            if self.charmap.get(c) is None:
                code = len(self.charmap)
                self.charmap[c] = code
                self.reverse_charmap[code] = c
        return s

    def report(self):
        print('CharmapBuilder: %d characters mapped' % len(self.charmap))

    def get_char(self, code):
        return self.reverse_charmap[code]

    def get_code(self, char):
        return self.charmap[char]

    def get_num_of_chars(self):
        return len(self.charmap)

    def get_charmap(self):
        return self.charmap