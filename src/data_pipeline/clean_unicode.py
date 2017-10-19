UNICODE_CONVERSION = {
    '\x81': ' ', #  (10)
    '\x89': ' ', #  (1)
    '\x92': "'", #  (4)
    '\x95': ' ', #  (1)
    '\x96': '-', #  (14)
    '\x97': '-', #  (4)
    '\xa0': ' ', #   (993)
    '¡': '!', # ¡ (8)
    '¤': 'x', # ¤ (2)
    '§': 'S', # § (23)
    '«': '<<', # « (1)
    '¬': '-', # ¬ (5)
    '\xad': '-', # ­ (28)
    '®': '(r)', # ® (8)
    '°': '°', # ° (19066)
    '±': '+-', # ± (1)
    '³': '^3', # ³ (1)
    '´': "'", # ´ (8)
    '·': "*", # · (13)
    '¹': "^1", # ¹ (1)
    'º': '°', # º (86)
    '»': '>>', # » (9)
    '¼': '1/4', # ¼ (37)
    '½': '1/2', # ½ (71)
    '¾': '3/4', # ¾ (16)
    '¿': '?', # ¿ (20)
    'À': 'A', # À (1)
    'Á': 'A', # Á (4)
    'É': 'E', # É (3)
    'Ñ': 'N', # Ñ (1)
    '×': 'x', # × (28)
    'Û': 'U', # Û (1)
    'Ü': 'U', # Ü (6)
    'ß': 'B', # ß (2)
    'à': 'a', # à (87)
    'á': 'a', # á (173)
    'â': 'a', # â (247)
    'ã': 'a', # ã (4)
    'ä': 'a', # ä (33)
    'å': 'a', # å (2)
    'ç': 'c', # ç (284)
    'è': 'e', # è (2167)
    'é': 'e', # é (13406)
    'ê': 'e', # ê (375)
    'ë': 'e', # ë (14)
    'ì': 'i', # ì (7)
    'í': 'i', # í (197)
    'î': 'i', # î (1103)
    'ï': 'i', # ï (122)
    'ñ': 'n', # ñ (2033)
    'ò': 'o', # ò (2)
    'ó': 'o', # ó (175)
    'ô': 'o', # ô (44)
    'õ': 'o', # õ (4)
    'ö': 'o', # ö (46)
    'ø': '0', # ø (4)
    'ù': 'u', # ù (121)
    'ú': 'u', # ú (94)
    'û': 'u', # û (164)
    'ü': 'u', # ü (44)
    'ÿ': 'y', # ÿ (1)
    'ź': 'z', # ź (1)
    'ˆ': '^', # ˆ (2)
    '˚': '°', # ˚ (34)
    '́': ' ', # ́ (13)
    '\u1ff0': ' ', # ῰ (1)
    '\u2002': ' ', #   (21)
    '\u2009': ' ', #   (13)
    '\u200b': ' ', # ​ (5)
    '‐': '-', # ‐ (10)
    '‑': '-', # ‑ (9)
    '‒': '-', # ‒ (1)
    '–': '-', # – (3497)
    '—': '-', # — (1030)
    '―': '-', # ― (2)
    '’': "'", # ’ (717)
    '‚': ',', # ‚ (65)
    '“': '"', # “ (44)
    '”': '"', # ” (73)
    '‟': '"', # ‟ (6)
    '•': '°', # • (235)
    '․': '.', # ․ (1)
    '…': '...', # … (3)
    '‧': ' ', # ‧ (5)
    '\u2028': ' ', #   (3)
    '\u2029': ' ', #   (55)
    '\u202a': ' ', # ‪ (1)
    '\u202d': ' ', # ‭ (8)
    '\u202e': ' ', # ‮ (3)
    '\u202f': ' ', #   (3)
    '‰': '%', # ‰ (3)
    '‱': '%', # ‱ (8)
    '″': '"', # ″ (4)
    '‹': ' ', # ‹ (1)
    '※': ' ', # ※ (2)
    '‿': '_', # ‿ (2)
    '⁄': '/', # ⁄ (23)
    '⅓': '1/3', # ⅓ (6)
    '⅔': '2/3', # ⅔ (2)
    '⅛': '1/8', # ⅛ (4)
    '⅞': '2/8', # ⅞ (1)
    '−': '-', # − (2)
    '◊': ' ', # ◊ (5)
    'ﬁ': 'fi', # ﬁ (4)
    'ﬂ': 'fl', # ﬂ (4)
    '�': ' ' # � (73)
}

def clean_unicode(s):
    for c in list(s):
        if UNICODE_CONVERSION.get(c):
            s = s.replace(c, UNICODE_CONVERSION[c])
    return s