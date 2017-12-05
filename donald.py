from itertools import accumulate
import gensim

class Trump:
    """
    Every tweet is a dictionary of the keys:
    "Date", "Time", "Tweet_Text", "Type", "Media_Type", "Hashtags", "Tweet_Id",
    "Tweet_Url", "twt_favourites_IS_THIS_LIKE_QUESTION_MARK", "Retweets".
    data source:
    https://www.kaggle.com/kingburrito666/better-donald-trump-tweets?email-verification=true&verification-id=1416547
    """

    _dataset = None

    def train():
        #return training set
        data = Trump._get_data()
        return Trump._partition(data, 'train')

    def val():
        #return validation set
        data = Trump._get_data()
        return Trump._partition(data, 'val')

    def test():
        #return test set
        data = Trump._get_data()
        return Trump._partition(data, 'test')

    def column(tweets, index="Tweet_Text"):
        #Return a vertical slice of the data. E.g. a list of only the text in all tweets, or a list of all tweet types.
        content_col = []
        for tweet in tweets:
            try: content_col.append(tweet[index])
            except:
                print("The indexkey was not found, or I could not process in tweet:", tweet)
        return content_col

    def _partition(data, part_name='train'):
        #This function returns a consistent part of the dataset

        def cumsum(lst):                                                                        #return a lists' cumulative list
            return list(accumulate(lst))
        def lzip(*lst):                                                                         # zip as list
            return list(zip(*lst))

        parts = [('_', 0.0), ('train', 0.8), ('val', 0.1), ('test', 0.1)]
        part_index = [idx for idx, pair in enumerate(parts) if pair[0] == part_name][0]         #grab index of chosen partition
        parts = lzip(lzip(*parts)[0], cumsum(lzip(*parts)[1]))                                  #unzip, take cumulative, rezip
        parts = [(pair[0], int(pair[1] * len(data))) for pair in parts]                         #normalize to number of datapoints

        return data[parts[part_index-1][1] : parts[part_index][1]]

    def _load_tweets():
        #returns a list of tweets (dictionaries). See help(Trump) for more information on tweets

        tweets = []
        with open('./data/Donald-Tweets.csv', 'r') as dataset:
            def process(txt):
                properties = txt.rstrip('\n').split(',', maxsplit=2)                            #split the first 2 properties
                txt = properties.pop()
                properties += txt.rstrip('\n').rsplit(',', maxsplit=7)                          #split the last 7 properties
                return properties

            labels = process(dataset.readline().rstrip(",,\n"))
            for line in dataset.read().split(",,\n"):
                tweet_contents = process(line)
                tweet = dict(zip(labels, tweet_contents))                                       #tweet becomes a dict for the labels
                tweets.append(tweet)
        return tweets[:-1]

    def _get_data():
        #getter for the full dataset
        if Trump._dataset is None:
            Trump._dataset = Trump._load_tweets()
        return Trump._dataset

class Processing:
    _w2v_model = None
    _keys = None
    # get_keras_embedding(train_embeddings=False) #useful for model?
    # Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations, 2014.
    #^ propose a cosine measure that amplifies short distance and reduces large. Used for analogy problems.

    def get(word):
        model = Processing.embeddings()
        return model.wv[word]

    def keys():
        if Processing._keys is None:
            Processing._keys = Processing.embeddings().vocab.keys()
        return Processing._keys

    def nearest(word, ntop=10):
        #returns the ntop nearest neighbours to a word, according to its embeddings
        model = Processing.embeddings()
        return model.wv.similar_by_word(word , topn = ntop)

    def similarity(w1, w2):
        #returns the cosine similarity between the given words
        model = Processing.embeddings()
        return model.wv.similarity(w1, w2)

    def nearest_tweets(tweet, tweets):
        #given a tweet(as a list of words) and a list of tweets, returns the nearest tweet in tweets.
        model = Processing.embeddings()
        similarities = []

        for twt in tweets:
            similarities.append(model.wv.n_similarity(tweet, twt))

        minsim = min(similarities)
        for idx, sim in enumerate(similarities):
            if sim == minsim:
                return tweets[idx]

    def embeddings():
        #getter for the full dataset
        print("init embed")
        if Processing._w2v_model is None:
            Processing._load_embeddings()
        return Processing._w2v_model

    def _load_embeddings():
        #Loads the word embeddings into memory
        print("please wait while I create a giant embeddingsmatrix..")
        Processing._w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
        print("I've loaded the word embeddings :)")
