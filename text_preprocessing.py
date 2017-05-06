import pandas as pd
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import csv
import time

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

t0 = time.time()
stops = [w for w in stopwords.words('english') if w not in ['when', 'where', 'why', 'how', 'what', 'which', 'who']]
lemmatizer = WordNetLemmatizer()

# train_data = pd.read_csv('../train.csv', index_col=0)
# # train_data = train_data[:50]
# for index, row in train_data.iterrows():
#     q1 = row[2]
#     q2 = row[3]
#     if pd.notnull(q1) and pd.notnull(q2):
#
#         tokens = nltk.word_tokenize(q1)
#         tagged = {x[0]:get_wordnet_pos(x[1]) for x in nltk.pos_tag(tokens)}
#         filtered_tokens = [token for token in tokens if token not in punctuation and token not in stops]
#         lemmas = [lemmatizer.lemmatize(token, tagged[token]) for token in filtered_tokens]
#         q1_prep = " ".join(lemmas)
#         train_data.set_value(row.name, 'question1', q1_prep)
#
#         tokens = nltk.word_tokenize(q2)
#         tagged = {x[0]: get_wordnet_pos(x[1]) for x in nltk.pos_tag(tokens)}
#         filtered_tokens = [token for token in tokens if token not in punctuation and token not in stops]
#         lemmas = [lemmatizer.lemmatize(token, tagged[token]) for token in filtered_tokens]
#         q2_prep = " ".join(lemmas)
#         train_data.set_value(row.name, 'question2', q2_prep)
#
# train_data.to_csv('train_preprocessed.csv', ',', columns=["qid1","qid2","question1","question2","is_duplicate"], quoting=csv.QUOTE_ALL)
#
# print(time.time() - t0, ' s')

test_data = pd.read_csv('../test.csv', index_col=0)
# test_data = test_data[:50]
for index, row in test_data.iterrows():
    q1 = row[0]
    q2 = row[1]
    if pd.notnull(q1) and pd.notnull(q2):

        tokens = nltk.word_tokenize(q1)
        tagged = {x[0]: get_wordnet_pos(x[1]) for x in nltk.pos_tag(tokens)}
        filtered_tokens = [token for token in tokens if token not in punctuation and token not in stops]
        lemmas = [lemmatizer.lemmatize(token, tagged[token]) for token in filtered_tokens]
        q1_prep = " ".join(lemmas)
        test_data.set_value(row.name, 'question1', q1_prep)

        tokens = nltk.word_tokenize(q2)
        tagged = {x[0]: get_wordnet_pos(x[1]) for x in nltk.pos_tag(tokens)}
        filtered_tokens = [token for token in tokens if token not in punctuation and token not in stops]
        lemmas = [lemmatizer.lemmatize(token, tagged[token]) for token in filtered_tokens]
        q2_prep = " ".join(lemmas)
        test_data.set_value(row.name, 'question2', q2_prep)

test_data.to_csv('test_preprocessed.csv', ',', columns=["question1", "question2"],
                  quoting=csv.QUOTE_ALL)

print(time.time() - t0, ' s')