import random

from datasets.stupid_recipes_dataset import StupidRecipesDataset
from model.one_hot_token_vectorizer import OneHotTokenVectorizer
from model.recipa_model import RecipaModel
from model.samples_builder import SamplesBuilder
from scrapers.epicurious import EpicusiourLoader
from data_pipeline.data_piptline import DataPipeline
from data_pipeline.clean_unicode import clean_unicode
from data_pipeline.clean_whitespace import clean_whitespace
from data_pipeline.charmap_builder import CharmapBuilder
from data_pipeline.one_hot_char_encode import OneHotCharEncoder
from data import special_tokens

# def append_chars(s, d):
#     for c in s:
#         d[ord(c)] = d.get(ord(c), 0) + 1


# data = EpicusiourLoader().load(r'c:\projects\recipa\data\epicurious', limit=1000)
sfd = StupidRecipesDataset()
data = [sfd.generate() for i in range(10000)]
samples = SamplesBuilder().build(data)
# for x in [(i, len([x[0] for x in samples if int(len(x[0])/100) == i])) for i in range(600)]:
#     print(x)
#
for sample in samples[:10]:
    print(''.join(sample[0]))
    print(''.join(sample[1]))
#
train, test = [samples[:int(len(samples) * 0.98)], samples[int(len(samples) * 0.98):]]
model = RecipaModel()
predictor = model.fit(train, test_samples=test)

# import _pickle
# _pickle.dump(d, open(r'c:\temp\d_after_cleanup.pickle', 'wb'))
# charmap_builder.report()
