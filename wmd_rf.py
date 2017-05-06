import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import time
import gensim

time_history = []
t0 = time.time()

# load data
print('Loading data...')
train_data = pd.read_csv('../train_preprocessed.csv', index_col=0)
test_data = pd.read_csv('../test_preprocessed.csv', index_col=0)
# train_data = train_data.head(10) # !!!
# test_data = test_data.head(10)
model = gensim.models.doc2vec.Doc2Vec.load('../enwiki_dbow/doc2vec.bin')

# print('Extracting training features...')
#
# labels_train = []
# features_train = []
# for index, row in train_data.iterrows():
#     q1 = row[2]
#     q2 = row[3]
#     label = row[4]
#     if pd.notnull(q1) and pd.notnull(q2):
#         q1 = q1.split(' ')
#         q2 = q2.split(' ')
#         feature = model.wmdistance(q1,q2)
#         if feature == float("inf") or not pd.notnull(feature):
#             feature = 10
#         features_train.append([feature])
#         labels_train.append(label)

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
    feature = model.wmdistance(q1, q2)
    if feature == float("inf") or not pd.notnull(feature):
        feature = 10
    features_test.append([feature])


# save extracted features
# pickle.dump(features_train, open('features_train_wmd.pkl', 'wb'))
# pickle.dump(labels_train, open('labels_train_wmd.pkl', 'wb'))
pickle.dump(features_test, open('features_test_wmd.pkl', 'wb'))

# train predicting model
# model = RandomForestRegressor()
# print('Training model...')
# model = model.fit(features_train, labels_train)
# pickle.dump(model, open('model_wmd_RF.pkl', 'wb'))
model = pickle.load(open('model_wmd_RF.pkl', 'rb'))

time_history.append(('training RF model', time.time() - t0))
t0 = time.time()

print('Predicting...')
predictions = model.predict(features_test)

time_history.append(('predicting', time.time() - t0))
t0 = time.time()

# save predictions to csv
print('Saving results...')
tmp = []
id = 0
for p in predictions:
    tmp.append({'test_id':id,'is_duplicate':p})
    id+=1

df = pd.DataFrame(tmp)
df.to_csv('labels_wmd_RF.csv', ',', columns=['test_id', 'is_duplicate'], index=False)

time_history.append(('saving predictions to csv file', time.time() - t0))
t0 = time.time()

# save CPU time history
text = ''
for t in time_history:
    text += str(t[0]) + ' ' + str(t[1]) + '\n'
with open('CPU_time_wmd_RF.txt', 'w') as file:
    file.write(text)