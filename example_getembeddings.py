from donald import Trump
import phrase2embedded as p2e

#get the validation set
validation = Trump.val()
validation_phrases = Trump.column(validation)

#define validation dataset embeddings
validation_embedded = p2e.convertphrases(validation_phrases, maxWords=45, padding=True)

#save validation dataset embeddings
p2e.save_embeddings(validation_embedded, "./data/embeddings_trumptweets_val.npz")

#load validation dataset embeddings
embds = p2e.load_embeddings(name="./data/embeddings_trumptweets_val.npz")

#sample some embeddings
print(embds[0].shape)
print(embds[67].shape)
print(embds[500].shape)

#free memory
del embds, validation, validation_phrases, validation_embedded
###################################################################################

#get the test set
test = Trump.test()
test_phrases = Trump.column(test)

#define test dataset embeddings
test_embedded = p2e.convertphrases(test_phrases, maxWords=45, padding=True)

#save validation dataset embeddings
p2e.save_embeddings(test_embedded, "./data/embeddings_trumptweets_test.npz")

#load validation dataset embeddings
embds = p2e.load_embeddings(name="./data/embeddings_trumptweets_test.npz")

#sample some embeddings
print(embds[0].shape)
print(embds[67].shape)
print(embds[500].shape)

#free memory
del embds, test, test_phrases, test_embedded
###################################################################################

#get the train set
train = Trump.train()
train_phrases = Trump.column(train)

#define train dataset embeddings
train_embedded = p2e.convertphrases(train_phrases, maxWords=45, padding=True)

#save train dataset embeddings
p2e.save_embeddings(train_embedded, "./data/embeddings_trumptweets_train.npz")

#load validation dataset embeddings
embds = p2e.load_embeddings(name="./data/embeddings_trumptweets_train.npz")

#sample some embeddings
print(embds[0].shape)
print(embds[67].shape)
print(embds[500].shape)

#free memory
del embds, train, train_phrases, train_embedded
###################################################################################
