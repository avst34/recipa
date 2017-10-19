import random

from data.recipe_record import RecipeRecord

FOODS = ['pizza', 'cake', 'hamburger', 'pasta', 'salad']
# FOODS = ['aa', 'ab', 'ac', 'ad']
# FOODS = ['1', '2', '3', '4']
# FOODS = ['a', 'b', 'c', 'd']


class StupidRecipesDataset:

    def __init__(self, generate_food_randomly=True, random_food_len=1):
        self.generate_food_randomly = generate_food_randomly
        self.random_food_len = random_food_len
        self.i = 0
        pass

    def get_food(self):
        if self.generate_food_randomly:
            food = ''.join([random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890') for i in range(self.random_food_len)])
        else:
            food = FOODS[self.i % len(FOODS)]
            self.i += 1
        return food

    def generate(self):
        food = self.get_food()

        return RecipeRecord(
            name='A recipe for %s' % food,
            # name='%s' % food,
            ingredients=[],
            instructions="Just make the %s, you'll do great" % food,
            # instructions="Just make the %s" % food,
            # instructions="%s" % food,
            # instructions="%s is good for you" % food,
            orig_record={}
        )