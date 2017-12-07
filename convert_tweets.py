from donald import Trump
from embeddings import Processing
from random import randint
import numpy as np
import nltk

training_data = Trump.train()
training_tweets = Trump.column(training_data, 'Tweet_Text')

def random_embedding():
    #Return random embedding. Possibly improve by weighing with frequency
    nwords = 3000000 - 1
    idx = randint(0, nwords)
    return Processing.get(Processing.embeddings().index2word[idx])

def save_tweets(tweets_embeddings):
    #save the dataset in a file that stores 7000 matrices,
    #which are concatenated word embeddings for its tweet contents
    #an empty array is added to prevent numpy from trying to infer fixed shape and crashing
    stitched_tweets = []
    stitched_tweets.append([np.zeros(2)] + [np.stack(tweet, axis=-1) for tweet in tweets_embeddings])

    np.savez('trump_embeddings', *stitched_tweets)

def load_tweets():
    #reads multiple tweet matrices from file.
    #returns a list of tweets, that are lists of word embeddings
    dataset = []

    loaded = np.load('trump_embeddings.npz')
    for tweet in loaded.files:
        #remove zero vector

        #print(loaded[tweet])

        content = [loaded[tweet]]
        if len(loaded[tweet].shape) > 1:
            content = np.split(loaded[tweet])
        dataset.append(content)
    return dataset

#tweets are ordered by latest first.
#The natural imput to the model should be earliest first.
#by popping the tweet stack we reverse the data in the process.
#new data format is a list of tweets, which are lists of embedding vectors
twts_embeds = []
nHits = nMiss = 0
for tweet in training_tweets:
  tweet_embedding = []
  for word in nltk.word_tokenize(tweet):
    #set embeddings for each word
    #if no word is found, insert a random embedding.

    embedding = None
    try:
      embedding = Processing.get(word)
      nHits += 1
    except:
      embedding = random_embedding()
      nMiss += 1
    tweet_embedding.append(embedding)

  twts_embeds.append(tweet_embedding)

print("# embeddings: ")
print("twts_embeds.len", len(twts_embeds))

save_tweets(twts_embeds)
print("saved tweets")
print("saved.shape", len(saved))

loaded = load_tweets()
print("loaded tweets: ")
print("loaded.shape", len(loaded))

print(nHits, nMiss)