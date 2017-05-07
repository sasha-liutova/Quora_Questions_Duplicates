import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import time
import gensim

# load data
print('Loading data...')
# train_data = pd.read_csv('../train_preprocessed.csv', index_col=0)
test_data = pd.read_csv('../test_preprocessed.csv', index_col=0)
# train_data = train_data.head(10) # !!!
# test_data = test_data.head(10)
model = gensim.models.word2vec.Word2Vec.load('../wiki_sg/word2vec.bin')

# print('Extracting training features...')
#
vector_length = len(model['the'])
# labels_train = []
# features_train = []
# for index, row in train_data.iterrows():
#     q1 = row[2]
#     q2 = row[3]
#     label = row[4]
#     if pd.notnull(q1) and pd.notnull(q2):
#         q1 = q1.split(' ')
#         q2 = q2.split(' ')
#         q1_vecs = [model[x] for x in q1 if x in model.vocab]
#         q2_vecs = [model[x] for x in q2 if x in model.vocab]
#         if len(q1_vecs) == 0:
#             q1_avg_vec = [0 for _ in range(vector_length)]
#         else:
#             q1_avg_vec = [sum([q1_vecs[j][i] for j in range(len(q1_vecs))])/len(q1_vecs) for i in range(vector_length)]
#         if len(q2_vecs) == 0:
#             q2_avg_vec = [0 for _ in range(vector_length)]
#         else:
#             q2_avg_vec = [sum([q2_vecs[j][i] for j in range(len(q2_vecs))])/len(q2_vecs) for i in range(vector_length)]
#         feature = np.concatenate((q1_avg_vec, q2_avg_vec), axis=0)
#         features_train.append(feature)
#         labels_train.append(label)
#
print('Extracting testing features...')

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
    q1_vecs = [model[x] for x in q1 if x in model.vocab]
    q2_vecs = [model[x] for x in q2 if x in model.vocab]
    if len(q1_vecs) == 0:
        q1_avg_vec = [0 for _ in range(vector_length)]
    else:
        # q1_avg_vec = [sum([q1_vecs[j][i] for j in range(len(q1_vecs))]) / len(q1_vecs) for i in range(vector_length)]
        q1_avg_vec = np.mean(np.array(q1_vecs), axis=0)
    if len(q2_vecs) == 0:
        q2_avg_vec = [0 for _ in range(vector_length)]
    else:
        # q2_avg_vec = [sum([q2_vecs[j][i] for j in range(len(q2_vecs))]) / len(q2_vecs) for i in range(vector_length)]
        q2_avg_vec = np.mean(np.array(q2_vecs), axis=0)
    feature = np.concatenate((q1_avg_vec, q2_avg_vec), axis=0)
    features_test.append(feature)


# save extracted features
# pickle.dump(features_train, open('features_train_word2vec.pkl', 'wb'))
# pickle.dump(labels_train, open('labels_train_word2vec.pkl', 'wb'))
np.save('features_test_word2vec.npy', features_test)


# features_train = pickle.load(open('./word2vec_results/features_train_word2vec.pkl', 'rb'))
# labels_train = pickle.load(open('./word2vec_results/labels_train_word2vec.pkl', 'rb'))
# features_test = np.load('./word2vec_results/features_test_word2vec.npy')

# # train predicting model
# t0 = time.time()
# model = RandomForestRegressor()
# print('Training model...')
# model = model.fit(features_train, labels_train)
# pickle.dump(model, open('model_word2vec_RF.pkl', 'wb'))
#
# print('Time: ', time.time() - t0)
# t0 = time.time()

model = pickle.load(open('./word2vec_results/model_word2vec_RF.pkl', 'rb'))

print('Predicting...')
predictions = model.predict(features_test)

# print('Time: ', time.time() - t0)
# t0 = time.time()

# save predictions to csv
print('Saving results...')
tmp = []
id = 0
for p in predictions:
    tmp.append({'test_id':id,'is_duplicate':p})
    id+=1

df = pd.DataFrame(tmp)
df.to_csv('labels_word2vec_RF.csv', ',', columns=['test_id', 'is_duplicate'], index=False)

# print('Time: ', time.time() - t0)
# t0 = time.time()
