import numpy as np
import nltk
import json
from nltk.tokenize import TweetTokenizer
import re
from tqdm import tqdm
import gensim


class Tweet_parser(object):

	def __init__(self, source='data'):
		self.word_embedding = gensim.models.KeyedVectors.load_word2vec_format('data/glove_word2vec.txt')
		self.vector_size = self.word_embedding.wv.vector_size
		print("VECTOR SIZE WORD:\t{}".format(self.vector_size))
		self.max_words = 35
		self.min_words = 10
		self.max_word_vec = self.max_words+5
		print("MAX WORDS:\t{}".format(self.max_words))

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
				return ["<hashtag>"] + self.parse_word(word[1:].lower()) + ["<allcaps>"]
			else:
				reparsed_word = []
				for split_word in re.sub( r"([A-Z])", r" \1", word[1:]).split():
					reparsed_word += self.parse_word(split_word)
				return ["<hashtag>"] + reparsed_word
		if len(word)>1:
			if all(s in "!?." for s in word):
				return re.sub( r"([.?!]){2,}", r" \1", word).split()+["<repeat>"]
		if any(i.isnumeric() for i in word) and not all(i.isalpha()for i in word):
			return ["<number>"]
		if  word.isupper() and len(word)>1:
			return  [word.lower(), "<allcaps>"]

		for i in range(len(word),-1,-1):
		#if it crashes, you might need to download corpuses --> nltk.download()
			if word[i-1]== word[-1]:
				break
			if word[:i] in nltk.corpus.words.words() or word[:i] in self.word_embedding.wv:
				if i == len(word)-1:
					return [word]
				else:
					return [word, "<elong>"]
		#special cases based on trump tweets
		if word == "Trump2016":
			return ["Trump", "2016"]
		if word == "Pence16":
			return ["Pence", "2016"]
		if "n't" == word[-3:]:
			if word.lower() == "can't":
				return ["can", "not"]
			if word.lower() == "won't":
				return ["will", "not"]
			return [word[:-3], "not"]
		if word == "I'm":
			return ["I","am"]
		if "'ll" in word:
			return [word[:-3], "will"]
		if word.lower() == "you're":
			return ["you", "are"]
		if "'ve" in word:
			return [word[:-3], "have"]
		if word == "ll":
			return ["will"]

		return [word]


	def sentence2embeddings(self, sentence):
		sentence_embedding = np.zeros([self.max_word_vec, self.vector_size])
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
			parsed = list(filter(("").__ne__, parsed))
			for i, parsed_word in enumerate(parsed):
				if i+embedding_idx >= self.max_word_vec:
					print("EXCEEDED MAX WORDS")
					return [None], False
				try:
					sentence_embedding[i+embedding_idx] = self.word_embedding.wv[parsed_word.lower()]
				except KeyError:
					# print("UNKNOWN WORD:\t{}".format(parsed_word))
					return [None], False
				embedding_idx += 1
			word_idx += 1

		return sentence_embedding, True
		

	def phrases2embeddings(self, phrases):
		phrases_embedding = np.zeros([len(phrases), self.max_word_vec, self.vector_size])
		fails = 0
		i = 0
		for sentence in tqdm(phrases):
			# print(' '.join(sentence))
			if len(sentence) <= self.max_words and len(sentence) >= self.min_words:
				sentence_embedding, passed = self.sentence2embeddings(sentence)
				if passed:
					phrases_embedding[i] = sentence_embedding
					i += 1
				else:
					fails += 1
		print("failed sentences:\t{}/{}".format(fails, len(phrases)))

		return phrases_embedding[:i]


	def sentence2embeddings_dynamic(self, sentence):
		sentence_embedding = []
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
			parsed = list(filter(("").__ne__, parsed))
			for i, parsed_word in enumerate(parsed):
				try:
					sentence_embedding.append(self.word_embedding.wv[parsed_word.lower()])
				except KeyError:
					# print("UNKNOWN WORD:\t{}".format(parsed_word))
					return [None], False
				embedding_idx += 1
			word_idx += 1

		return np.array(sentence_embedding), True

	def phrases2embeddings_dynamic(self, phrases):
		phrases_embedding = [[] for i in range(50)]
		fails = 0
		for sentence in tqdm(phrases):
			# print(' '.join(sentence))
			
			sentence_embedding, passed = self.sentence2embeddings_dynamic(sentence)
			if passed:
				idx = sentence_embedding.shape[0]
				if idx < 50:
					phrases_embedding[idx].append(sentence_embedding)
			else:
				fails += 1
		print("failed sentences:\t{}/{}".format(fails, len(phrases)))
		for i, emb_dim in enumerate(phrases_embedding):
			phrases_embedding[i] = np.array(emb_dim)

		phrases_embedding = np.asarray(phrases_embedding)
		return phrases_embedding

	def sentence2embeddings_size(self, sentence):
		embedding_idx = 0
		word_idx = 0
		failure_case = {}

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
			parsed = list(filter(("").__ne__, parsed))
			for i, parsed_word in enumerate(parsed):
				if parsed_word.lower() in self.word_embedding.wv:
					embedding_idx+=1
				else:
					if parsed_word in failure_case:
						failure_case[parsed_word] += 1
					else:
						failure_case.update({parsed_word:1})
			word_idx += 1

		return embedding_idx, failure_case

	def phrases2embeddings_size(self, phrases):
		fails = 0
		i = 0
		histogram = np.zeros([100])
		failure_cases = {}
		for sentence in tqdm(phrases):
			length, failure_case = self.sentence2embeddings_size(sentence)
			if len(failure_case) > 0:
				for key, value in failure_case.items():
					if key in failure_cases:
						failure_cases[key] += value
					else:
						failure_cases.update({key:value})
			else:
				histogram[length] += 1
		failures = sorted(failure_cases.items(), key=lambda item: (item[1], item[0]))[::-1][:100]
		print(failures)
		print(histogram)
		return histogram, failures

	def load_tweets(self, file):
		# with open(file, encoding="utf8") as f:
		# 	x = json.load(f)
		x = np.load(file)
		tweett = TweetTokenizer()
		x_ = []
		for phrase in x:
			phrase = tweett.tokenize(phrase)
			if len(phrase) <= self.max_words:
				x_.append(phrase)
				
		y = [len(z) for z in x_]
		print("NR OF TWEETS:\t{}".format(len(x)))
		print("MEAN LENGTH SENTENCE:\t{}".format(np.mean(y)))
		print("STD LENGTH SENTENCE:\t{}".format(np.std(y)))
		return x_


if __name__ == '__main__' :
	tp = Tweet_parser()
	x = tp.load_tweets('data/usa_geo_tokens.npy')
	y = tp.phrases2embeddings_dynamic(x)
	# y = tp.phrases2embeddings(x)

	np.save('data/geo_embedding_dynamic.npy', y)