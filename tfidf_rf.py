import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import time

time_history = []
t0 = time.time()
# load data
print('Loading data...')
train_data = pd.read_csv('../train_preprocessed.csv', index_col=0)
test_data = pd.read_csv('../test_preprocessed.csv', index_col=0)
# train_data = train_data.head(10) # !!!
# test_data = test_data.head(10)
print('Training data shape: ', train_data.shape)
print('Testing data shape: ', test_data.shape)

q_1_dup, q_1_diff, q_2_dup, q_2_diff, q_all = [], [], [], [], []
for index, row in train_data.iterrows():
    q1 = row[2]
    q2 = row[3]
    if pd.notnull(q1) and pd.notnull(q2):
        q_all.append(q1)
        q_all.append(q2)
        label = row[4]
        if label == 1:
            q_1_dup.append(q1)
            q_2_dup.append(q2)
        elif label == 0:
            q_1_diff.append(q1)
            q_2_diff.append(q2)

q1_test, q2_test = [], []
for index, row in test_data.iterrows():
    q1 = row[0]
    q2 = row[1]
    q1_test.append(q1)
    q2_test.append(q2)

time_history.append(('feature extraction', time.time() - t0))
t0 = time.time()

print('Fitting vectorizer...')
# feature extraction
vectorizer = TfidfVectorizer()
vectorizer.fit(q_all)

time_history.append(('fitting vectorizer', time.time() - t0))
t0 = time.time()

print('Vectorizing training data...')
#       from training data
q_1_dup_trans = vectorizer.transform(q_1_dup).toarray()
q_2_dup_trans = vectorizer.transform(q_2_dup).toarray()
q_1_diff_trans = vectorizer.transform(q_1_diff).toarray()
q_2_diff_trans = vectorizer.transform(q_2_diff).toarray()
q_pairs_dup = [np.concatenate((x, y)) for x,y in zip(q_1_dup_trans, q_2_dup_trans)]
q_pairs_diff = [np.concatenate((x, y)) for x,y in zip(q_1_diff_trans, q_2_diff_trans)]
features_train = np.concatenate((q_pairs_dup, q_pairs_diff))
labels_train = [1 for x in range(len(q_pairs_dup))]
labels_train.extend([0 for y in range(len(q_pairs_diff))])

time_history.append(('vectorizing training data', time.time() - t0))
t0 = time.time()

#       from testing data
print('Vectorizing testing data...')
q1_test_trans = vectorizer.transform(q1_test).toarray()
q2_test_trans = vectorizer.transform(q2_test).toarray()
features_test = [np.concatenate((x, y)) for x,y in zip(q1_test_trans, q2_test_trans)]

time_history.append(('vectorizing testing data', time.time() - t0))
t0 = time.time()

# save extracted features
pickle.dump(features_train, open('features_train_tfidf.pkl', 'wb'))
pickle.dump(labels_train, open('labels_train_tfidf.pkl', 'wb'))
pickle.dump(features_test, open('features_test_tfidf.pkl', 'wb'))

# train predicting model
model = RandomForestRegressor()
print('Training model...')
model = model.fit(features_train, labels_train)
pickle.dump(model, open('model_tfidf_RF.pkl', 'wb'))

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
df.to_csv('labels_tfidf_RF.csv', ',', columns=['test_id', 'is_duplicate'], index=False)

time_history.append(('saving predictions to csv file', time.time() - t0))
t0 = time.time()

# save CPU time history
text = ''
for t in time_history:
    text += str(t[0]) + ' ' + str(t[1]) + '\n'
with open('CPU_time_tfidf_RF.txt', 'w') as file:
    file.write(text)