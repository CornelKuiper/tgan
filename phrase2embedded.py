from embeddings import Processing
from random import randint
import numpy as np
import nltk
from pathlib import Path

def save_embeddings(tweets_embeddings, name, vocab = []):
    #save the output of convertphrases in a npz file. Each phrase is stored as a matrix of dim x nWords in the file.
    #an empty array is prepended to prevent numpy from trying to infer fixed shape and crashing
    stitched_tweets = [np.zeros(2)] + [np.stack(tweet, axis=-1) for tweet in tweets_embeddings]

    if vocab == []:
        np.savez(name, *stitched_tweets)
    else
        #setup word -> embedding dictionary
        phrase2embedding['dummy'] = stitched_tweets[0]
        for idx in range(1, len(vocab)):
            phrase2embedding[vocab[idx]] = stitched_tweets[idx]

        np.savez(name, **phrase2embedding)



def load_embeddings(name="./data/embeddings_trumptweets_train.npz", maxWords=45, padding=False):
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

def convertphrases(listofphrases):
    #convert a list of phrases (a list of strings/sentences) to a list of lists of embeddings
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
