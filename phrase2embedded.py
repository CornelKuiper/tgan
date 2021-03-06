from embeddings import Processing
from random import randint
import numpy as np
import nltk
from pathlib import Path

def save_embeddings(tweets_embeddings, name, vocab = []):
    #save the output of convertphrases in a npz file. Each phrase is stored as a matrix of dim x nWords in the file.
    #an empty array is prepended to prevent numpy from trying to infer fixed shape and crashing
    stitched_tweets = [np.zeros(2)] + [np.stack(tweet, axis=-1) for tweet in tweets_embeddings if len(tweet) > 0]

    if vocab == []:
        np.savez(name, *stitched_tweets)
    else:
        #setup word -> embedding dictionary
        phrase2embedding['dummy'] = stitched_tweets[0]
        for idx in range(1, len(vocab)):
            phrase2embedding[vocab[idx]] = stitched_tweets[idx]

        np.savez(name, **phrase2embedding)

def load_padded_tweet_embeddings():
    return np.load("./data/tweet_embeddings.npy")

def load_embeddings(name="./data/embeddings_trumptweets_train.npz", maxWords=40, padding=False):
    #reads multiple phrase matrices from file.
    #returns a vector of phrases, that are matrices of 300xnWords
    #if nonzero maxWords will discard all phrases that have more words than maxWords
    #if padding is set to True, all phrases are appended with zero vectors until all phrases contain maxWords.
    print("Loading embeddings from: ", name, "with a maximum of ", maxWords, " words. \
    Padding of shorter sentences is set to ", padding, ".")

    if not Path(name).exists():
        url = "https://www.dropbox.com/sh/faoxnfui3ndo26k/AABrpK3opNF9d96dJZ3xVcy-a?dl=0"
        print("Given embeddings file was not found. Maybe you can find it at: ", url)
        print("Make sure the file is located in the given path")
        return

    #load data
    loaded = np.load(name)                                  # close to avoid leaking file descriptor
    content = loaded[loaded.files[0]][1:]                   # get data, strip empty matrix

    #throw away outlier length sentences
    if maxWords:
        mask = [phrase.shape[1] <= maxWords for phrase in content]
        content = content[mask]

    #when set, pad phrases up to maxWords. (phrase.shape[0] == dimensionality)
    if padding:
        for phrase_idx in range(content.shape[0]):
            phrase = content[phrase_idx]
            dim = phrase.shape[0]
            nWords = phrase.shape[1]

            content[phrase_idx] = np.concatenate([phrase, np.zeros((dim, maxWords - nWords))], axis=1)

    print("Loading embeddings complete.")
    return content

def asMatrix(loadembeddingsoutput):
    #Converts an array of matrices to a big matrix, assuming matrix shapes to be consistent
    #The array dimensions will be the zeroth axis of the output.
    #The order of matrix axes is preserved, and come after the zeroth axis.
    matrix = np.stack(loadembeddingsoutput, axis=-1)                                #concat array matrices
    matrix = np.swapaxes(matrix, 0, 2)                                              #set array dimensions as 0-axis
    return np.swapaxes(matrix, 1, 2)                                                #preserve secondary axes

def asTokens(listofphrases):
    #Given a list of phrases (strings) return a list of phrases (list of words)
    return [nltk.word_tokenize(phrase) for phrase in listofphrases]

def convertphrases(listofphrases, useGlove = False):
    """
    convert a list of phrases (a list of words) to a list of lists of embeddings
    #the result is preferably saved with save_embeddings and loaded similarly

    for cornells tweet dataset, pass argument: list(np.load("D:/ML_datasets/tweet_tokens.npy"))
    for trump twitter data, pass argument: asTokens(Trump.column(Trump.train()))
    """

    def random_embedding():
        #Return random embedding. Possibly improve by weighing with frequency
        nwords = 3000000 - 1
        idx = randint(0, nwords)
        return Processing.get(Processing.embeddings().index2word[idx])


    def parse_advanced(word):
        if "https" in word or "www" in word:
            return ["<URL>"]
        if "/" in word:
            return word.split("/")
        if word[0] == "@":
            return ["<USER>"]
        if "<3" in word:
            return ["<HEART>"]
        if all(i.isdigit() for i in word) or all(i[1:].isdigit() for i in word):
            return ["<NUMBER>"]
        if word[0] == "#":
            if word[1:].isupper():
                return ["<HASHTAG>", word[1:].lower(), "<ALLCAPS>"]
            else:
                return ["<HASHTAG>"] + re.sub( r"([A-Z])", r" \1", word[1:]).split()
        if any(s in "!?." for s in word):
            return re.sub( r"([.?!]){2,}", r" \1", x).split()+["<REPEAT>"]
        if  word.isupper():
            return  [word.lower(), "<ALLCAPS>"]
        for i in range(len(word),0,-1):
            #if it crashes, you might need to download corpuses --> nltk.download()
            if word[:i] in nltk.corpus.words.words():
                if i == len(word):
                    return [word]
                else:
                    return [word, "<ELONG>"]
        return [False]
        #original also included emotes check

        

    def parse(word):
        #this should be consistent with special symbols used in embeddings such as <number>, <hashtag>
        #currently; words in ignore return False
        #otherwise, the correct word (s) is returned, belong to word
        ignorelist = ['"', ',', '.', '!', '?', ':', 'of', 'and', 'a', '/', 'to', '(', ')', '...']

        if word[0] == '#' or word[0] == '@':
            return [word[1:]]
        if word not in ignorelist:
            return [word]
        if "http" in word:
            return [word]

        return [False]

    def setEmbedding(word):
        word_embedding = None
        hitsmiss = [0,0]
        try:
            word_embedding = Processing.get(word)
            hitsmiss[0] += 1
        except:
            #if no word is found, insert a random embedding.
            word_embedding = random_embedding()
            hitsmiss[1] += 1
        return (word_embedding, hitsmiss)

    embedded_phrases = np.zeros([len(listofphrases), 45, 300])
    hitsnmisses = {'hits':0, 'miss': 0}
    for phrase in listofphrases:
        phrase_embedding = np.zeros([45, 300])
        idx = 0
        while idx < len(phrase):
            token = phrase[idx]
            if token in ".?!":
                while phrase[idx+1] in ".?!" and idx < len(phrase):
                    token += phrase[idx+1]
                    idx += 1
            if token == "@":
                token += phrase[idx+1]
                idx += 1

            words = parse(token)                      #parse word

            for word in words:
                if not word: continue                   #ignore useless

                (word_embedding, hitsmiss) = setEmbedding(word)
                phrase_embedding.append(word_embedding)
                hitsnmisses['hits'] = hitsmiss[0]
                hitsnmisses['miss'] = hitsmiss[1]
        embedded_phrases.append(phrase_embedding)
    print("Done finding the embeddings of all phrases & their words")
    print("During the search, I found ", hitsnmisses['hits'], " and was unable to embed ", hitsnmisses['miss'], " words.")
    print("Random words have been inserted where no such embedding exists.")
    return embedded_phrases
