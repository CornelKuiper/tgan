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
        with open('./data/Donald-Tweets.csv', 'r', encoding='utf8') as dataset:
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
