    ignorelist = ['"', ',', '.', '!', '?', ':', 'of', 'and', 'a', '/', 'to', '(', ')', '...']
    containlist = ['http']

    embedded_phrases = []
    nHits = nMiss = 0
    for phrase in listofphrases:
      phrase_embedding = []
      for word in nltk.word_tokenize(phrase):
        if word[0] == '#' or word[0] == '@':
            word = word[1:]
        if word in ignorelist:
            continue
        if "http" in word:
            continue