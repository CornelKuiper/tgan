from donald import Trump, Processing

#load tweet text content. (note, tweet dates,times and hashtags may be as relevant)
training_data = Trump.train()
training_tweets = Trump.column(training_data, 'Tweet_Text')		#grab a feature of all tweets

Processing._load_embeddings()


