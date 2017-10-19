from data_pipeline.clean_unicode import clean_unicode
from data_pipeline.clean_whitespace import clean_whitespace
from data import special_tokens
from utils import lists_sum

class SamplesBuilder:

    def __init__(self):
        pass

    def build(self, records):
        samples = []

        for rec in records:

            x_tokens = special_tokens.wrap_start_end_tokens(
                # lists_sum([special_tokens.wrap_ingredient_tokens(list(clean_whitespace(clean_unicode(ing)))) for ing in rec.ingredients])
                [] + special_tokens.wrap_instructions_tokens(list(clean_whitespace(clean_unicode(rec.instructions))))
            )

            y_tokens = special_tokens.wrap_start_end_tokens(
                list(clean_whitespace(clean_unicode(rec.name)))
            )

            samples.append(
                (
                    x_tokens, y_tokens
                )
            )

        return samples

