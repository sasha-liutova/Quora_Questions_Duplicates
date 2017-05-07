import pandas as pd
import pickle
import time
import numpy as np
import gensim
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

print('Loading data...')

features_train = pickle.load(open('./word2vec_results/features_train_word2vec.pkl', 'rb'))
labels_train = pickle.load(open('./word2vec_results/labels_train_word2vec.pkl', 'rb'))
features_test = np.load('./word2vec_results/features_test_word2vec.npy')

# features_train = features_train[:1000]
# labels_train = labels_train[:1000]

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

best_params = {'C': 1, 'epsilon': 0.4, 'kernel': 'linear'}

t0 = time.time()

print('Training model...')

model = SVR(C=best_params['C'], epsilon=best_params['epsilon'], kernel=best_params['kernel'])
model = model.fit(features_train, labels_train)
pickle.dump(model, open('model_word2vec_svm.pkl', 'wb'))

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
df.to_csv('labels_word2vec_svm.csv', ',', columns=['test_id', 'is_duplicate'], index=False)

print('Time: ', time.time() - t0)
t0 = time.time()