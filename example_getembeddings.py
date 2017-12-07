from donald import Trump
import phrase2embedded

embds = phrase2embedded.load_embeddings(name="./data/trump_embeddings.npz")

print(embds[0])
print(embds[1])

#embed the validation set
validation = Trump.val()
validation_phrases = Trump.column(validation)

validation_embedded = phrase2embedded.convertphrases(validation_phrases)
phrase2embedded.save_embeddings(validation_embedded, "./data/trump_validation_embedded")
