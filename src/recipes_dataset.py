import json

class Sample:

    def __init__(self, title, content, category, site, url):
        self.title = title
        self.content = content
        self.category = category
        self.site = site
        self.url = url

class RecipiesDataset:

    def __init__(self, data):
        self._raw_data = data

    @staticmethod
    def load_from_json(json_path):
        with open(json_path, 'rb') as f:
            return RecipiesDataset(json.load(f))

