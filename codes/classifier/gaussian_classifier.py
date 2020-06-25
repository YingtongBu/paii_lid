import os
import sys
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import svm
from sklearn import metrics


'https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html'

def read_ivector(filepath):
  utt2lang = {}
  with open('data/train_all3/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
  with open('data/lre07_all3/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
  X = []
  y = []
  files = os.listdir(filepath)
  for file in files:
    with open(f'{filepath}/{file}', 'r') as f:
      for line in f:
        item = line.strip().split()
        utt = item[0]
        label = utt2lang[utt]
        if label == 'us':
          label = 0
        elif label == 'uk':
          label = 1
        else:
          label = 2
        vector = [float(x) for x in item[1:][1:-1]]
        X.append(vector)
        y.append(label)
  return X, y

def read_xvector(filepath):
  X = []
  y = []
  files = os.listdir(filepath)
  for file in files:
    with open(f'{filepath}/{file}', 'r') as f:
      for line in f:
        item = line.strip().split()
        utt = item[0]
        label = utt.split('-')[0]
        if label == 'us':
          label = 0
        elif label == 'uk':
          label = 1
        else:
          continue
        vector = [float(x) for x in item[1:][1:-1]]
        X.append(vector)
        y.append(label)
  return X, y

def posteriors(model_name, X):
  loaded_model = pickle.load(open(model_name, 'rb'))
  posteriors = loaded_model.predict_proba(X)
  p = [list(x) for x in list(posteriors)]
  with open('posteriors', 'w') as f:
    f.writelines(str(line) + '\n' for line in p)

  # with open('gold', 'w') as f:
  #   f.writelines(str(line) + '\n' for line in y_test)

if __name__ == '__main__':
  # nj = int(sys.argv[1])
  # save_model = str(sys.argv[2])
  # train_dir = str(sys.argv[3])
  # test_dir = str(sys.argv[4])
  X_train, y_train = read_xvector('exp_2/xvectors_train')
  X_test, y_test = read_xvector('exp_2/xvectors_test')
  kernel = 1.0 * RBF(1.0)
  print('start training...')

  # gpc = GaussianProcessClassifier(
  #   kernel=kernel, random_state=0, n_jobs=nj).fit(X_train, y_train)
  # gpc = GaussianProcessClassifier(
  #   kernel=kernel, random_state=0).fit(X_test, y_test)
  # pickle.dump(gpc, open(model_name, 'wb'))
  # print('start evaluating...')
  # print(gpc.score(X_train, y_train))
  # print(gpc.score(X_test, y_test))

  clf = svm.SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
  clf.fit(X_train, y_train)
  clf_predictions = clf.predict(X_test)
  print("Accuracy: {}%".format(metrics.accuracy_score(y_test, clf_predictions)))


'''
sudo python3 gpc.py 10 gpc.sav

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
'''