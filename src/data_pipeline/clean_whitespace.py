def clean_whitespace(s):
    s = s.replace('\y', ' ')
    while True:
        new_s = s.replace('  ', ' ')
        if len(new_s) == len(s):
            break
        s = new_s
    return s