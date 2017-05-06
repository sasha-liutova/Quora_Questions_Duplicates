import pandas as pd
import pickle
import time
import gensim
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

print('Loading data...')
features_train = pickle.load(open('./wmd_rf_results/features_train_wmd.pkl', 'wb'))
labels_train = pickle.load(open('./wmd_rf_results/labels_train_wmd.pkl', 'wb'))
features_test = pickle.load(open('./wmd_rf_results/features_test_wmd.pkl', 'wb'))

print('Tuning model hyperparameters...')
t0 = time.time()

parameters = {'C': [1, 5, 10], 'epsilon': [0.1, 0.2, 0.3]}
svr = SVR()
grid_search = GridSearchCV(svr, parameters)
grid_search.fit(features_train, labels_train)
print('Grid Search Results:')
print('best_params: ', grid_search.best_params_)
print(grid_search.cv_results_)
best_params = grid_search.best_params_

print('Time: ', time.time() - t0)
t0 = time.time()

print('Training model...')

model = SVR(C=best_params['C'], epsilon=best_params['epsilon'], kernel='rbf')
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