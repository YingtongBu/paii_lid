import os
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def read_ivectors(filepath):
  utt2lang = {}
  utt2nid = {}
  unique_nids = []
  with open('data/utt2nid', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nid = item[-1]
      utt2nid[utt] = nid
  with open('mturk/utt2lang6.v3', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
      if utt2nid[utt] not in unique_nids:
        unique_nids.append(utt2nid[utt])
  X = []
  y = []
  label_dict = {'us': 0, 'uk': 1, 'nnn': 2, 'nnl': 3, 'nnm': 4, 'nnh': 5}
  files = os.listdir(filepath)
  for file in files:
    with open(f'{filepath}/{file}', 'r') as f:
      for line in f:
        item = line.strip().split()
        utt = item[0]
        label = utt2lang[utt]
        vector = [float(x) for x in item[1:][1:-1]]
        one_pos = unique_nids.index(utt2nid[utt])
        dummy_list = [0] * len(unique_nids)
        dummy_list[one_pos] = 1
        try:
          cls = label_dict[label]
        except KeyError:
          continue
        X.append(vector + dummy_list)
        # X.append(vector)
        y.append(cls)
  print(len(X), len(y))
  return X, y



def score(y_predict, y_gold):
  score = 0
  for i in range(len(y_predict)):
    if y_predict[i] == y_gold[i]:
      score += 1
  print(score)
  print(len(y_gold))
  return score/len(y_gold)

if __name__ == '__main__':
  X, y = read_ivectors('results/ivectors.data_mturk_train3535')
  test_X, test_y = read_ivectors('results/ivectors.test_mturk_v3')

  norm = Normalizer()
  X_normed = np.array(norm.fit_transform(X))
  test_X_normed = np.array(norm.transform(test_X))
  logreg = LogisticRegression().fit(X_normed, y)

  #rfe = RFE(logreg, 20).fit(os_data_X, os_data_y.values.ravel())
  # print(reg.score(X_normed, y))
  # print(test_X[0])
  # print(reg.predict(np.array([test_X[0]])))
  # print(test_y[0])
  y_predict = logreg.predict(np.array(test_X_normed))
  print(score(y_predict, test_y))
  print(score(logreg.predict(np.array(X_normed)), y))

  plt.plot(test_y)
  plt.show()
