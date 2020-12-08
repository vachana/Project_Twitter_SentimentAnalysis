import csv
import os
import re
import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.tokenize import TreebankWordTokenizer
data = []
# preprocessing
csv_file = open('training.csv', encoding='latin-1')
csv_reader = csv.reader(csv_file, delimiter=',')

for row in csv_reader:
    data.append((row[-1], row[0]))
noun = 0.0
adjective = 0.0
adverb = 0.0
pronoun = 0.0
with open("resultset.csv", 'a') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in data:
        noun = noun + len([token for token, pos in pos_tag(word_tokenize(i[0])) if pos.startswith('N')])
        adjective = adjective + len([token for token, pos in pos_tag(word_tokenize(i[0])) if pos.startswith('JJ')])
        adverb = adverb + len([token for token, pos in pos_tag(word_tokenize(i[0])) if pos.startswith('RB')])
        pronoun = pronoun + len([token for token, pos in pos_tag(word_tokenize(i[0])) if pos.startswith('PRP')])
        tokenize = TreebankWordTokenizer()
        words = (tokenize.tokenize(i[0]))
        print(noun)
        def readwords(filename):
            filename = os.path.join(filename)
            f = open(filename)
            words = [line.rstrip() for line in f.readlines()]
            return words
        positive = readwords('positive-words.txt')
        negative = readwords('negative-words.txt')

        pos = 0
        neg = 0

        for key in words:
            if key in positive:
                pos += 1
            if key in negative:
                neg += 1
        print("pos",pos)
        nouncount = noun / (len(words) * 1.0)
        adjectivecount = adjective / (len(words) * 1.0)
        pronouncount = pronoun / (len(words) * 1.0)
        adverbcount = adverb / (len(words) * 1.0)

        mylist= [nouncount,adjectivecount,pronouncount,adverbcount,pos,neg,i[1]]
        wr.writerow(mylist)
