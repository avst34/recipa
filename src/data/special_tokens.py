START_TEXT = '<START>'
END_TEXT = '<END>'
START_INGREDIENTS = '<START_INGREDIENTS>'
END_INGREDIENTS = '<END_INGREDIENTS>'
START_INGREDIENT = '<START_INGREDIENT>'
END_INGREDIENT = '<END_INGREDIENT>'
START_INSTRUCTIONS = '<START_INSTRUCTIONS>'
END_INSTRUCTIONS = '<END_INSTRUCTIONS>'

def wrap_start_end_tokens(tokens):
    return [START_TEXT] + tokens + [END_TEXT]

def wrap_instructions_tokens(tokens):
    return [START_INSTRUCTIONS] + tokens + [END_INSTRUCTIONS]

def wrap_ingredient_tokens(tokens):
    return [START_INGREDIENT] + tokens + [END_INGREDIENT]

def wrap_ingredients_tokens(tokens):
    return [START_INGREDIENTS] + tokens + [END_INGREDIENTS]

