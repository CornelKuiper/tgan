from donald import Trump, Processing

#load tweet text content. (note, tweet dates,times and hashtags may be as relevant)
training_data = Trump.train()
training_tweets = Trump.column(training_data, 'Tweet_Text')		#grab a feature of all tweets

# Processing.embeddings()
print("getting magicalplumbus: ")
print(Processing.get("theplumbuseveryonehasone"))

# print("nearest to king: ")
# print(Processing.nearest('king', 5))
#
# print("similarity between king and queen: ")
# print(Processing.similarity('king', 'queen'))
#
# print("nearest tweet to 'we must build a wall'")
# print(Processing.nearest_tweets(['we must build a wall'], ["we shall make mexico pay for the wall", "apple pies are delicious"]))
