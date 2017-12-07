import numpy as np
import requests
from pathlib import Path

def load_tweets():
    #reads multiple tweet matrices from file.
    #returns a npvector of tweets, that are wordembedding matrices of shape (300, nWords)

    dataset = []
    tweetembeddings = Path("./trump_embeddings.npz")

    if not tweetembeddings.exists():
        url = "https://www.dropbox.com/sh/faoxnfui3ndo26k/AABrpK3opNF9d96dJZ3xVcy-a?dl=0"
        print("please download the tweet embeddings from: ", url)
        return

    loaded = np.load('trump_embeddings.npz')
    for tweet in loaded.files:
        #still to remove zero vector

        content = [loaded[tweet]]
        if len(loaded[tweet].shape) > 1:
            content = np.split(loaded[tweet])
        dataset.append(content)

    print(tweetvector.shape)
    print(tweetvector[0].shape)

    return dataset[0][0]
