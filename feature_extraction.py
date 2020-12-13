import csv
import os
import nltk
from nltk import word_tokenize, pos_tag
import math
from nltk.corpus import stopwords
import itertools
from numba import cuda
from nltk.tokenize import TreebankWordTokenizer


def readwords(filename):
    filename = os.path.join(filename)
    f = open(filename)
    words = [line.rstrip() for line in f.readlines()]
    return words

positives = readwords('positive-words.txt')
negatives = readwords('negative-words.txt')

data = []
# preprocessing

# csv_file = open('testdata.csv', encoding='latin-1')  ######################################test file
csv_file = open('training.csv', encoding='latin-1')  ######################################trainning file
csv_reader = csv.reader(csv_file, delimiter=',')

for row in itertools.islice(csv_reader, 100000):
    p = []
    stop = set(stopwords.words('english'))
    for i in row[-1].split():
        if i not in stop:
            p.append(i)
    data.append((p, row[0]))
num_rows = len(data)

all_words = []
for (words, sentiment) in data:
    all_words.extend(words)

print("D_freq_list started ...")
D_freq_list = nltk.FreqDist(all_words)
print("D_freq_list finished ...")

count_idf = {}
for row in data:
    words = set([word for word in row[0]])
    for word in words:
        if word not in count_idf:
            count_idf[word] = 1
        else:
            count_idf[word] += 1

print("start writing features...")
with open("feature_training.csv", 'w+') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i, row in enumerate(data):
        if i%10000 == 0:
            print(i, " done...")
        postag = pos_tag(row[0])
        d_freq = nltk.FreqDist(row[0])

        positive = 0.0
        negative = 0.0
        noun = 0.0
        adjective = 0.0
        adverb = 0.0
        pronoun = 0.0
        emotioncount = 0.0

        for token, pos in postag:
            tf = math.log10(1 + d_freq[token])
            idf = math.log10(num_rows / count_idf[token])
            tfidf = tf * idf
            emotioncount += token.count('!')
            if token in positives:
                positive += tfidf
            if token in negatives:
                negative += tfidf
            if pos.startswith('N'):
                noun += tfidf
            if pos.startswith('JJ'):
                adjective += tfidf
            if pos.startswith('RB'):
                adverb += tfidf
            if pos.startswith('PRP'):
                pronoun += tfidf

        l = len(row[0])
        if l>0:
            nouncount = noun / l
            adjectivecount = adjective / l
            pronouncount = pronoun / l
            adverbcount = adverb / l

        mylist= [nouncount,adjectivecount,pronouncount,adverbcount,positive,negative,emotioncount,row[1]]
        wr.writerow(mylist)

