def tweets_geo_usa():
    phrases = []
    with open('./data/tweets_geo_usa.csv', 'r', encoding='utf8') as dataset:
        for line in dataset.read().split(',en\n'):
            phrases.append(line)

    return phrases

def tweets_geo_uk():
    phrases = []
    with open('./data/tweets_geo_uk.csv', 'r', encoding='utf8') as dataset:
        for line in dataset.read().split(',en\n'):
            phrases.append(line)

    return phrases
