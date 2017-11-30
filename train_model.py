from donald import Trump, Processing

#load tweet text content. (note, tweet dates,times and hashtags may be as relevant)
training_data = Trump.train()
training_tweets = Trump.column(training_data, 'Tweet_Text')		#grab a feature of all tweets

Processing.embeddings()
Processing.get('king')
Processing.nearest('king', 5)
Processing.similarity('king', 'queen')
Processing.nearest_tweets(['we must build a wall'], ['we shall make mexico pay for the wall'])
