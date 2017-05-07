import pickle
import pandas as pd
import numpy as np
import gensim

test_data = pd.read_csv('../test_preprocessed.csv', index_col=0)
# test_data = test_data.head(100)
model = gensim.models.doc2vec.Doc2Vec.load('../enwiki_dbow/doc2vec.bin')

print('Extracting testing doc2vec features...')

features_test = []

for index, row in test_data.iterrows():
    q1 = row[0]
    q2 = row[1]
    if pd.notnull(q1) and pd.notnull(q2):
        q1 = q1.split(' ')
        q2 = q2.split(' ')
    elif not pd.notnull(q1):
        q1 = [' ']
    elif not pd.notnull(q2):
        q2 = [' ']
    q1_vec = model.infer_vector(q1)
    q2_vec = model.infer_vector(q2)
    feature = np.concatenate((q1_vec, q2_vec), axis=0)
    features_test.append(feature)

print('Saving data...')
np.save('features_test_doc2vec.npy', features_test)

print('Trying to load data from file...')
check = np.load('features_test_doc2vec.npy')
print(check[:2])
