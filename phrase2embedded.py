from embeddings import Processing
from random import randint
import numpy as np
import nltk
from pathlib import Path

def save_embeddings(tweets_embeddings, name):
    #save the output of convertphrases in a npz file. Each phrase is stored as a matrix of dim x nWords in the file.
    #an empty array is prepended to prevent numpy from trying to infer fixed shape and crashing
    stitched_tweets = []
    stitched_tweets.append([np.zeros(2)] + [np.stack(tweet, axis=-1) for tweet in tweets_embeddings])

    np.savez('./data/' + name, *stitched_tweets)

def load_embeddings(name="./data/trump_embeddings.npz"):
    #reads multiple phrase matrices from file.
    #returns a vector of phrases, that are matrices of 300xnWords
    dataset = []

    if not Path(name).exists():
        url = "https://www.dropbox.com/sh/faoxnfui3ndo26k/AABrpK3opNF9d96dJZ3xVcy-a?dl=0"
        print("Given embeddings file was not found. Maybe you can find it at: ", url)
        return

    loaded = np.load(name)
    for phrase in loaded.files:
        content = [loaded[phrase]]
        if len(loaded[phrase].shape) > 1:
            #resplit phrase into words
            content = np.split(loaded[phrase])
        dataset.append(content)

    print("Loading embeddings complete")
    return dataset[0][0][1:]

def convertphrases(listofphrases):
    #convert a list of phrases to a list of lists of embeddings
    #the result is preferably saved with save_embeddings and loaded similarly

    def random_embedding():
        #Return random embedding. Possibly improve by weighing with frequency
        nwords = 3000000 - 1
        idx = randint(0, nwords)
        return Processing.get(Processing.embeddings().index2word[idx])

    embedded_phrases = []
    nHits = nMiss = 0
    for phrase in listofphrases:
      phrase_embedding = []
      for word in nltk.word_tokenize(phrase):
        #set embeddings for each word
        #if no word is found, insert a random embedding.

        word_embedding = None
        try:
          word_embedding = Processing.get(word)
          nHits += 1
        except:
          word_embedding = random_embedding()
          nMiss += 1
        phrase_embedding.append(word_embedding)

      embedded_phrases.append(phrase_embedding)
    print("Done finding the embeddings of all phrases & their words")
    print("During the search, I found ", nHits, " and was unable to embed ", nMiss, " words.")
    print("Random words have been inserted where no such embedding exists.")
    return embedded_phrases
