import pandas as pd
import pickle
import time
import gensim
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

print('Loading data...')

features_train = pickle.load(open('./wmd_results/features_train_wmd.pkl', 'rb'))
labels_train = pickle.load(open('./wmd_results/labels_train_wmd.pkl', 'rb'))

test_data = pd.read_csv('../test_preprocessed.csv', index_col=0)

model = gensim.models.doc2vec.Doc2Vec.load('../enwiki_dbow/doc2vec.bin')

# features_train = features_train[:1000]
# labels_train = labels_train[:1000]

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

np.save('features_test_wmd.npy', features_test)

# print('Tuning model hyperparameters...')
# t0 = time.time()
#
# parameters = {'C': [1], 'epsilon': [0.3], 'kernel':['linear', 'poly', 'rbf']}
# svr = SVR()
# grid_search = GridSearchCV(svr, parameters)
# grid_search.fit(features_train, labels_train)
# print('Grid Search Results:')
# print('best_params: ', grid_search.best_params_)
# print(grid_search.cv_results_)
# best_params = grid_search.best_params_

best_params = {'C': 1, 'epsilon': 0.3, 'kernel':'rbf'}

t0 = time.time()

print('Training model...')

model = LinearSVR(C=best_params['C'], epsilon=best_params['epsilon'])
model = model.fit(features_train, labels_train)
pickle.dump(model, open('model_wmd_svm.pkl', 'wb'))

print('Time: ', time.time() - t0)
t0 = time.time()

print('Predicting...')
predictions = model.predict(features_test)

print('Time: ', time.time() - t0)
t0 = time.time()

# save predictions to csv
print('Saving results...')
tmp = []
id = 0
for p in predictions:
    tmp.append({'test_id':id,'is_duplicate':p})
    id+=1

df = pd.DataFrame(tmp)
df.to_csv('labels_wmd_svm.csv', ',', columns=['test_id', 'is_duplicate'], index=False)

print('Time: ', time.time() - t0)
t0 = time.time()