import requests
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from data.recipe_record import  RecipeRecord

class EpicuriusScraper:

    def __init__(self):
        pass

    def _request(self, page_num, parse_json=True):
        r = requests.get('https://www.epicurious.com/search', params={
            'content': 'recipe',
            'page': page_num,
            'xhr': 'true'
        }, headers={'x-requested-with': 'XMLHttpRequest'})
        if r.status_code != 200:
            raise Exception('Request status code:' + str(r.status_code))
        if parse_json:
            return json.loads(r.text)
        else:
            return r.text.encode('utf8')

    def get_pages_count(self):
        return self._request(1)['page']['totalCount']

    def scrape(self, dest_folder):
        n_pages = self.get_pages_count()
        print('Fetching %d pages' % n_pages)
        def scrape_page(page_num):
            r = self._request(page_num, parse_json=False)
            with open(os.path.join(dest_folder, '%d.json' % page_num), 'wb') as out_f:
                out_f.write(r)
            print('fetched page %d' % page_num)
            time.sleep(0.3)

        with ThreadPoolExecutor(10) as tpe:
            list(tpe.map(scrape_page, range(1, n_pages + 1)))

class EpicusiourLoader:

    def normalize(self, recipe_record):
        required = [recipe_record.get('hed'), recipe_record.get('prepSteps'), recipe_record.get('ingredients')]
        if not all(required):
            return None
        return RecipeRecord(name=recipe_record['hed'],
                            instructions=' '.join(recipe_record['prepSteps']),
                            ingredients=recipe_record['ingredients'],
                            orig_record=recipe_record)

    def load(self, dir, limit=None):
        data = []
        for f_path in os.listdir(dir):
            with open(os.path.join(dir, f_path), 'rb') as f:
                data.extend(
                    [self.normalize(x) for x in json.load(f)['items']]
                )
                if limit is not None and len(data) >= limit:
                    data = data[:limit]
                    break
        data = [x for x in data if x]
        print('Loaded %d items after filtering' % len(data))
        return data
