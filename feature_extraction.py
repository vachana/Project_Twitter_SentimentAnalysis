import csv
import os
import nltk
from nltk import word_tokenize, pos_tag
import math
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

# for row in itertools.islice(csv_reader, 100000):
pos_counter = 0
neg_counter = 0
ntr_counter = 0

for row in csv_reader:
    if row[0] == "4" and pos_counter < 50000:
        data.append((word_tokenize(row[-1]), row[0]))
        pos_counter += 1
    elif row[0] == "2" and ntr_counter < 50000:
        data.append((word_tokenize(row[-1]), row[0]))
        ntr_counter += 1
    elif neg_counter < 50000:
        data.append((word_tokenize(row[-1]), row[0]))
        neg_counter += 1

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
with open("features_training.csv", 'w+') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i, row in enumerate(data):
        if i%10000 == 0:
            print(i, " done...")
        postag = pos_tag(row[0])
        d_freq = nltk.FreqDist(row[0])

        positive_words = 0.0
        negative_words = 0.0
        noun = 0.0
        adjective = 0.0
        adverb = 0.0
        pronoun = 0.0

        for token, pos in postag:
            tf = math.log10(1 + d_freq[token])
            idf = math.log10(num_rows / count_idf[token])
            tfidf = tf * idf

            if token in positives:
                positive_words += tfidf
            if token in negatives:
                negative_words += tfidf
            if pos.startswith('N'):
                noun += tfidf
            if pos.startswith('JJ'):
                adjective += tfidf
            if pos.startswith('RB'):
                adverb += tfidf
            if pos.startswith('PRP'):
                pronoun += tfidf

        l = len(row[0])
        nouncount = noun / l
        adjectivecount = adjective / l
        pronouncount = pronoun / l
        adverbcount = adverb / l

        mylist= [nouncount,adjectivecount,pronouncount,adverbcount,positive_words,negative_words,row[1]]
        wr.writerow(mylist)
# (x1-y1)^2 + (x2-y2)^2 + …
