import pandas as pd
import numpy as np
import pickle
import time
import gensim
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

t0 = time.time()

# load data
print('Loading data...')
# train_data = pd.read_csv('../train_preprocessed.csv', index_col=0)
test_data = pd.read_csv('../test_preprocessed.csv', index_col=0)
model = gensim.models.doc2vec.Doc2Vec.load('../enwiki_dbow/doc2vec.bin')

features_train = pickle.load(open('./doc2vec_results_1/features_train_doc2vec.pkl', 'rb'))
labels_train = pickle.load(open('./doc2vec_results_1/labels_train_doc2vec.pkl', 'rb'))
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
    q1_vec = model.infer_vector(q1)
    q2_vec = model.infer_vector(q2)
    feature = np.concatenate((q1_vec, q2_vec), axis=0)
    features_test.append(feature)


# print('Tuning model hyperparameters...')
# t0 = time.time()
#
# parameters = {'C': [1], 'epsilon': [0.2, 0.4], 'kernel':['linear']}
# svr = SVR()
# grid_search = GridSearchCV(svr, parameters)
# grid_search.fit(features_train, labels_train)
# print('Grid Search Results:')
# print('best_params: ', grid_search.best_params_)
# print(grid_search.cv_results_)
# best_params = grid_search.best_params_

best_params = {'C': 1, 'epsilon': 0.4, 'kernel': 'linear'}


print('Time: ', time.time() - t0)
t0 = time.time()

print('Training model...')

model = SVR(C = best_params['C'], epsilon=best_params['epsilon'], kernel=best_params['kernel'])
model = model.fit(features_train, labels_train)
pickle.dump(model, open('model_doc2vec_svm.pkl', 'wb'))

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
df.to_csv('labels_doc2vec_svm.csv', ',', columns=['test_id', 'is_duplicate'], index=False)

print('Time: ', time.time() - t0)
t0 = time.time()
