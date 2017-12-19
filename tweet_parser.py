import numpy as np
import nltk
import json
from nltk.tokenize import TweetTokenizer
import re
from tqdm import tqdm
from tweet_parser import Processing
class Tweet_parser(object):

	def __init__(self, source='data'):
		self.embedding_list = np.load("{}/embedding.npy".format(source))
		self.word_list = np.load("{}/words.npy".format(source))
		self.vector_size = self.embedding_list.shape[-1]
		self.max_words = 45
		print("MAX WORDS:\t{}".format(self.max_words))
		print("VECTOR SIZE WORD:\t{}".format(self.vector_size))

	def parse_word(self, word):
		if "http" in word or "www" in word:
			return ["<url>"]
		if "/" in word:
			return word.split("/")
		if word[0] == "@":
			return ["<user>"]
		if "<3" in word:
			return ["<heart>"]
		if word[0] == "#":
			if word[1:].isupper():
				return ["<hashtag>", word[1:].lower(), "<allcaps>"]
			else:
				return ["<hashtag>"] + re.sub( r"([A-Z])", r" \1", word[1:]).split()
		if all(s in "!?." for s in word) and len(word)>1:
			return re.sub( r"([.?!]){2,}", r" \1", word).split()+["<repeat>"]
		if any(i.isnumeric() for i in word) and not all(i.isalpha()for i in word):
			return ["<number>"]
		if  word.isupper() and len(word)>1:
			return  [word.lower(), "<allcaps>"]
		for i in range(len(word),-1,-1):
		#if it crashes, you might need to download corpuses --> nltk.download()
			if word[i-1]== word[-1]:
				break
			if word[:i] in nltk.corpus.words.words():
				if i == len(word)-1:
					return [word]
				else:
					return [word, "<elong>"]
		return [word]


	def sentence2embeddings(self, sentence):
		sentence_embedding = np.zeros([self.max_words, self.vector_size])
		embedding_idx = 0
		word_idx = 0

		while word_idx < len(sentence):
			word = sentence[word_idx]
			#requires look ahead
			if word in ".?!":
				while word_idx < len(sentence)-1:
					if sentence[word_idx+1] in ".?!":
						word += sentence[word_idx+1]
						word_idx += 1
					else:
						break
			if word == u"\U0001F1FA":
				if sentence[word_idx+1] ==  u"\U0001F1F8":
					word_idx += 1
					word = "USA"
			parsed = self.parse_word(word)

			for i, parsed_word in enumerate(parsed):
				if i+embedding_idx >= self.max_words:
					print("EXCEEDED MAX WORDS")
					return [None], False

				index = np.where(self.word_list==parsed_word)[0]
				if len(index) == 0:
					if not parsed_word.islower():
						parsed_word = parsed_word.lower()
						index = np.where(self.word_list==parsed_word)[0]
						if len(index) == 0:
							# print("UNKNOWN WORD:\t{}".format(parsed_word))
							return [None], False
					else:
						return [None], False
				# print(parsed_word, end=' ')
				sentence_embedding[i+embedding_idx] = self.embedding_list[index]
				embedding_idx += 1
			word_idx += 1

		return sentence_embedding, True

	def phrases2embeddings(self, phrases):
		phrases_embedding = np.zeros([len(phrases), self.max_words, self.vector_size])
		fails = 0
		i = 0
		for sentence in tqdm(phrases):
			# print(' '.join(sentence))
			if len(sentence) <= self.max_words:
				sentence_embedding, passed = self.sentence2embeddings(sentence)
				if passed:
					phrases_embedding[i] = sentence_embedding
					i += 1
				else:
					fails += 1
		print("failed sentences:\t{}/{}".format(fails, len(phrases)))

		return phrases_embedding

	def load_tweets(self, file):
		with open(file, encoding="utf8") as f:
			x = json.load(f)

		tweett = TweetTokenizer()
		x = [tweett.tokenize(phrase['text']) for phrase in x]
		y = [len(z) for z in x]
		print("NR OF TWEETS:\t{}".format(len(x)))
		print("MEAN LENGTH SENTENCE:\t{}".format(np.mean(y)))
		print("STD LENGTH SENTENCE:\t{}".format(np.std(y)))
		embeddings = self.phrases2embeddings(x)
		return embeddings

if __name__ == '__main__' :
	tp = Tweet_parser()
	x = tp.load_tweets('data/trump_tweets.json')
	np.save('data/trump_embedding.npy', x)