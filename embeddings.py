import gensim

class Processing:
    "This class interfaces with the google word2vec embeddings. It is desirable to only load necessary data, though."

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
        if Processing._w2v_model is None:
            Processing._load_embeddings()
        return Processing._w2v_model

    def _load_embeddings():
        #Loads the word embeddings into memory
        print("please wait while I create a giant embeddingsmatrix..")
        Processing._w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
        print("I've loaded the word embeddings :)")
