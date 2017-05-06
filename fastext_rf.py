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
train_data = pd.read_csv('./train_preprocessed.csv', index_col=0)
test_data = pd.read_csv('./test_preprocessed.csv', index_col=0)
# train_data = train_data.head(10) # !!!
# test_data = test_data.head(10)

vectors_lib = pd.read_table('./wiki.en.vec')
# vectors_lib = pd.read_table('../wiki.en.vec', nrows=100) # !!!
vectors_dict = {}
for index, row in vectors_lib.iterrows():
    a = row[0].split(' ')
    vectors_dict[a[0]] = a[1:]
vector_length = len(vectors_dict['the']) -1

print('Extracting training features...')

labels_train = list(train_data['is_duplicate'])
features_train = []

for index, row in train_data.iterrows():
    q1 = row[2]
    q2 = row[3]
    if pd.notnull(q1) and pd.notnull(q2):

        q1_vec = [0 for x in range(vector_length)]
        for word in q1.split(' '):
            word = word.lower()
            if word in vectors_dict:
                word_vec = vectors_dict[word]
                # print('Word ', word, ' in question ', index, ' found.')
            else:
                word_vec = [0 for x in range(vector_length)]
            for i in range(vector_length):
                q1_vec[i] += float(word_vec[i])
        q1_vec = [x/len(q1.split(' ')) for x in q1_vec]

        q2_vec = [0 for x in range(vector_length)]
        for word in q2.split(' '):
            word = word.lower()
            if word in vectors_dict:
                word_vec = vectors_dict[word]
            else:
                word_vec = [0 for x in range(vector_length)]
            for i in range(vector_length):
                q2_vec[i] += float(word_vec[i])
            q2_vec = [x / len(q2.split(' ')) for x in q2_vec]

        feature = q1_vec
        feature.extend(q2_vec)

        features_train.append(feature)

print('Extracting testing features...')

features_test = []

for index, row in test_data.iterrows():
    q1 = row[0]
    q2 = row[1]
    if pd.notnull(q1) and pd.notnull(q2):

        q1_vec = [0 for x in range(vector_length)]
        for word in q1.split(' '):
            word = word.lower()
            if word in vectors_dict:
                word_vec = vectors_dict[word]
            else:
                word_vec = [0 for x in range(vector_length)]
            for i in range(vector_length):
                q1_vec[i] += float(word_vec[i])
        q1_vec = [x / len(q1.split(' ')) for x in q1_vec]

        q2_vec = [0 for x in range(vector_length)]
        for word in q2.split(' '):
            word = word.lower()
            if word in vectors_dict:
                word_vec = vectors_dict[word]
            else:
                word_vec = [0 for x in range(vector_length)]
            for i in range(vector_length):
                q2_vec[i] += float(word_vec[i])
            q2_vec = [x / len(q2.split(' ')) for x in q2_vec]

        feature = q1_vec
        feature.extend(q2_vec)

        features_test.append(feature)

# save extracted features
pickle.dump(features_train, open('features_train_fasttext.pkl', 'wb'))
pickle.dump(labels_train, open('labels_train_fasttext.pkl', 'wb'))
pickle.dump(features_test, open('features_test_fasttext.pkl', 'wb'))

# train predicting model
model = RandomForestRegressor()
print('Training model...')
model = model.fit(features_train, labels_train)
pickle.dump(model, open('model_fasttext_RF.pkl', 'wb'))

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
df.to_csv('labels_fasttext_RF.csv', ',', columns=['test_id', 'is_duplicate'], index=False)

time_history.append(('saving predictions to csv file', time.time() - t0))
t0 = time.time()

# save CPU time history
text = ''
for t in time_history:
    text += str(t[0]) + ' ' + str(t[1]) + '\n'
with open('CPU_time_fasttext_RF.txt', 'w') as file:
    file.write(text)