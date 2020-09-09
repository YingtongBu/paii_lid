import os
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def read_ivectors(filepath):
  utt2lang = {}
  with open('mturk/utt2lang6.v3', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
  X = []
  y = []
  label_dict = {'us': -2, 'uk': -1, 'nnn': 0, 'nnl': 1, 'nnm': 2, 'nnh': 3}
  files = os.listdir(filepath)
  # files = [file for file in os.listdir(filepath) if 'train' in file]
  for file in files:
    with open(f'{filepath}/{file}', 'r') as f:
      for line in f:
        item = line.strip().split()
        utt = item[0]
        label = utt2lang[utt]
        vector = [float(x) for x in item[1:][1:-1]]
        try:
          cls = label_dict[label]
        except KeyError:
          continue
        X.append(vector)
        y.append(cls)
  print(len(X), len(y))
  return X, y

def reg2cls(y_predict):
  y_predict_cls = []
  for y in y_predict:
    if y < -2.5:
      cls = -2
    elif y >= 3.5:
      cls = 3
    else:
      cls = round(y)
    y_predict_cls.append(cls)
  return y_predict_cls

def score(y_predict, y_gold):
  score = 0
  for i in range(len(y_predict)):
    if y_predict[i] == y_gold[i]:
      score += 1
  return score/len(y_gold)

if __name__ == '__main__':
  X, y = read_ivectors('results/ivectors.data_mturk_v2')
  test_X, test_y = read_ivectors('results/ivectors.test_mturk_v2')

  norm = Normalizer()
  X_normed = np.array(norm.fit_transform(X))
  test_X_normed = np.array(norm.transform(test_X))
  reg = LinearRegression().fit(X_normed, y)
  # print(reg.score(X_normed, y))
  # print(test_X[0])
  # print(reg.predict(np.array([test_X[0]])))
  # print(test_y[0])
  y_predict = reg.predict(np.array(test_X_normed))
  print(score(reg2cls(y_predict), test_y))
  plt.plot(reg2cls(y_predict))
  plt.plot(test_y)
  plt.show()