import pickle
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# train_data = pd.read_csv('../train.csv', index_col=0)
features_train = pickle.load(open('./doc2vec_results_1/features_train_doc2vec.pkl', 'rb'))
# labels = pickle.load(open('./doc2vec_results_1/labels_train_doc2vec.pkl', 'rb'))
model = pickle.load(open('./doc2vec_results_1/model_doc2vec_RF.pkl', 'rb'))
predictions = model.predict(features_train)

# # calculating TOP and LOW 50 questions
# differences = [abs(x-y) for x,y in zip(labels, predictions)]
# diff_sorted = sorted(((value, index) for index, value in enumerate(differences)))
# print(diff_sorted)
# top50_indexes = diff_sorted[:50]
# worst50_indexes = diff_sorted[-50:]
# top50 = [(train_data.loc[x[1], 'question1'], train_data.loc[x[1], 'question2'], x[0], labels[x[1]], predictions[x[1]]) for x in top50_indexes]
# worst50 = [(train_data.loc[x[1], 'question1'], train_data.loc[x[1], 'question2'], x[0], labels[x[1]], predictions[x[1]]) for x in worst50_indexes]
#
# print('TOP')
# for q in top50:
#     print(q)
# print('-------------------------')
# print('WORST')
# for q in worst50:
#     print(q)

# analyzing predictions distribution
n, bins, patches = plt.hist(predictions, 10, facecolor='blue')
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Distribution of precitions')
# plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()