from itertools import accumulate

class Trump:
    """
    Every tweet is a dictionary of the keys:
    "Date", "Time", "Tweet_Text", "Type", "Media_Type", "Hashtags", "Tweet_Id",
    "Tweet_Url", "twt_favourites_IS_THIS_LIKE_QUESTION_MARK", "Retweets".
    data source:
    https://www.kaggle.com/kingburrito666/better-donald-trump-tweets?email-verification=true&verification-id=1416547
    """

    def train():
        #return training set
        data = Trump.__load_tweets__()
        return Trump.__partition__(data, 'train')

    def val():
        #return validation set
        data = Trump.__load_tweets__()
        return Trump.__partition__(data, 'val')

    def test():
        #return test set
        data = Trump.__load_tweets__()
        return Trump.__partition__(data, 'test')

    def column(tweets, index="Tweet_Text"):
        #Return a vertical slice of the data. E.g. a list of only the text in all tweets, or a list of all tweet types.
        content_col = []
        for tweet in tweets:
            try: content_col.append(tweet[index])
            except:
                print("The indexkey was not found, or I could not process in tweet:", tweet)
        return content_col

    def __partition__(data, part_name='train'):
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

    def __load_tweets__():
        #returns a list of tweets (dictionaries). See help(Trump) for more information on tweets

        tweets = []
        with open('Donald-Tweets.csv', 'r') as dataset:
            def process(txt): return txt.rstrip('\n').split(',')
            labels = process(dataset.readline())

            for line in dataset.read().split(",,\n"):
                tweet_contents = process(line)
                tweet = dict(zip(labels, tweet_contents))
                tweets.append(tweet)
        return tweets[:-1]
