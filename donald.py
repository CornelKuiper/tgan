import csv

class Trump:
    """
    Every tweet is a dictionary of the keys:
    "Date", "Time", "Tweet_Text", "Type", "Media_Type", "Hashtags", "Tweet_Id",
    "Tweet_Url", "twt_favourites_IS_THIS_LIKE_QUESTION_MARK", "Retweets".
    data source:
    https://www.kaggle.com/kingburrito666/better-donald-trump-tweets?email-verification=true&verification-id=1416547
    """

    def load_tweets():
        "returns a list of tweets (dictionaries). See help(Trump) for more information on tweets."

        tweets = []
        with open('Donald-Tweets.csv', 'r') as dataset:
            def process(txt): return txt.rstrip('\n').split(',')
            labels = process(dataset.readline())

            for line in dataset.read().split(",,\n"):
                tweet_contents = process(line)
                tweet = dict(zip(labels, tweet_contents))
                tweets.append(tweet)
        return tweets[:-1]

    def column(tweets, index="Tweet_Text"):
        "Return a vertical slice of the data. E.g. a list of only the text in all tweets, or a list of all tweet types."
        content_col = []
        for tweet in tweets:
            try: content_col.append(tweet[index])
            except:
                print("The indexkey was not found, or I could not process in tweet:", tweet)
        return content_col
