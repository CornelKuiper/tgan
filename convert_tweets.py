from donald import Trump, Processing
from random import randint
import numpy as np

training_data = Trump.train()
training_tweets = Trump.column(training_data, 'Tweet_Text')

def random_embedding():
    #Return random embedding. Possibly improve by weighing with frequency
    nwords = 3000000 - 1
    idx = randint(0, nwords)
    return Processing.get(Processing.embeddings().index2word[idx])

def save_tweets(tweets_embeddings):
    for tweet in tweets_embeddings:
        for embd in tweet:
            try:
                print(embd.shape)
            except:
                print(embd)
    #save the dataset in a file that stores 7000 matrices,
    #which are concatenated word embeddings for its tweet contents
    stitched_tweets = []
    stitched_tweets.append([np.stack(tweet, axis=-1) for tweet in tweets_embeddings])

    np.savez('trump_embeddings', *stitched_tweets)

def load_tweets():
    #reads multiple tweet matrices from file.
    #returns a list of tweets, that are lists of word embeddings
    dataset = []

    loaded = np.load('trump_embeddings.npz')
    for tweet in loaded.files:
        content = np.split(tweet)
        dataset.append(content)
    return dataset

#tweets are ordered by latest first.
#The natural imput to the model should be earliest first.
#by popping the tweet stack we reverse the data in the process.
#new data format is a list of tweets, which are lists of embedding vectors
twts_embeds = []

nHits = 0
nMiss = 0
for tweet in training_tweets:
    tweet_embedding = []
    for word in tweet.split(' '):
        embedding = None
        try:
            embedding = Processing.get(word)
            nHits += 1
        except:
            #if no word is found, insert a random embedding?
            embedding = random_embedding()
            nMiss += 1

        tweet_embedding.append(embedding)
        print("gotten word ", word)
    twts_embeds.append(tweet_embedding)

#print("tweets0", twts_embeds[0][:10])
#print("tweets1", twts_embeds[1][:10])
print("tweets00", twts_embeds[0][0][:10])
print("tweets11", twts_embeds[1][1][:10])

save_tweets(twts_embeds)
print("saved tweets")

foo = load_tweets()
print("loaded tweets: ")
#print("loadedtweets0", foo[0][:10])
#print("loadedtweets1", foo[1][:10])
print("loadedtweets00", foo[0][0][:10])
print("loadedtweets11", foo[1][1][:10])

print(nHits, nMiss)
