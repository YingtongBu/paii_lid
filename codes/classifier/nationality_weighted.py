import os
import numpy as np
from scipy.optimize import leastsq, minimize
from scipy.special import xlogy
from pa_nlp.measure import Measure
import math

def nid2prob():
  utt2lang = {}
  nid2prob = {}
  label_dict = {'us': 0, 'uk': 1, 'nnn': 2, 'nnl': 3, 'nnm': 4, 'nnh': 5}
  with open('mturk/utt2lang6.v5', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if lang == 'Others':
        continue
      utt2lang[utt] = lang
  with open('data/utt2nid', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nid = item[-1]
      if nid == 'Taiwan,ProvinceofChina':
        nid = 'Taiwan(ProvinceofChina)'
      try:
        lang_index = label_dict[utt2lang[utt]]
      except KeyError:
        continue
      if nid not in nid2prob:
        nid2prob[nid] = [0]*6
      nid2prob[nid][lang_index] += 1
  to_write = []
  for nid in nid2prob:
    summ = sum(nid2prob[nid])
    for i in range(len(label_dict)):
      nid2prob[nid][i] = str(round(nid2prob[nid][i]/summ, 4))
    to_write.append(nid + ' ' + str(summ) + ' ' + nid2prob[nid][0]
                    + ' ' + nid2prob[nid][1] + ' ' + nid2prob[nid][2]
                    + ' ' + nid2prob[nid][3] + ' ' + nid2prob[nid][4]
                    + ' ' + nid2prob[nid][5] + '\n')
    print(nid, nid2prob[nid])
  with open('data/nid2prob.v5', 'w') as f:
    f.writelines(line for line in to_write)
  return nid2prob

def read_priors_posteriors(posteriors_filepath, nation_num):
  utt2lang = {}
  utt2nid = {}
  nid2summ = {}
  nid2prob = {}
  with open('data/utt2nid', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nid = item[-1]
      utt2nid[utt] = nid
  with open('data/nid2prob.v5', 'r') as f:
    for line in f:
      item = line.strip().split()
      nid = item[0]
      summ = item[1]
      if int(summ) < nation_num:
        continue
      nid2summ[nid] = summ
      nid2prob[nid] = []
      for i in range(2, 8):
        nid2prob[nid].append(float(item[i]))
  with open('mturk/utt2lang6.v5', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang

  priors = []
  posteriors = []
  y = []
  label_dict = {'us': 0, 'uk': 1, 'nnn': 2, 'nnl': 3, 'nnm': 4, 'nnh': 5}
  with open(posteriors_filepath, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      label = label_dict[utt2lang[utt]]
      y.append(label)
      posterior = [float(x) for x in item[1:][1:-1]]
      posteriors.append(posterior)
      try:
        prior = nid2prob[utt2nid[utt]]
        priors.append(prior)
      except KeyError:
        priors.append(posterior)

  print(len(priors), len(posteriors), len(y))
  return priors, posteriors, y

def fixed_weight(w, priors, posteriors, y_gold):
  if len(priors) == len(posteriors) == len(y_gold):
    y_pred = []
    y_goldd = []
    correct = 0
    for prior, posterior, y in zip(priors, posteriors, y_gold):
      # print(prior, posterior)
      if isinstance(w, list):
        p = [w_ * x + (1-w_) * y for (w_, x, y) in zip(w, prior, posterior)]
      else:
        p = [w * x + (1-w) * y for (x, y) in zip(prior, posterior)]
      logistic_regression = posterior.index(max(posterior))
      nationality = prior.index(max(prior))
      cls = p.index(max(p))
      # print(cls, y, nationality, logistic_regression)
      # if cls == y:
      if cls in [3,4,5]:
        cls = 2
      if y in [2]:
        continue
      if y in [3,4,5]:
        y = 2
      if cls == y:
        correct += 1
      y_pred.append(cls)
      y_goldd.append(y)
    result = Measure.calc_precision_recall_fvalue(y_goldd, y_pred)
    print(result)
    return correct/len(priors)
  return False

# one case split as six samples if nid sum larger than nation_num
def make_features(posteriors_filepath, nation_num, train=True):
  utt2lang = {}
  utt2nid = {}
  nid2summ = {}
  nid2prob = {}
  with open('data/utt2nid', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nid = item[-1]
      utt2nid[utt] = nid
  with open('data/nid2prob.v5', 'r') as f:
    for line in f:
      item = line.strip().split()
      nid = item[0]
      summ = item[1]
      if int(summ) < nation_num:
        continue
      nid2summ[nid] = summ
      nid2prob[nid] = []
      for i in range(2, 8):
        nid2prob[nid].append(float(item[i]))
  with open('mturk/utt2lang6.v5', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
  X = []
  y = []
  label_dict = {'us': 0, 'uk': 1, 'nnn': 2, 'nnl': 3, 'nnm': 4, 'nnh': 5}
  with open(posteriors_filepath, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      if utt2nid[utt] not in nid2prob and train:
        continue

      label = label_dict[utt2lang[utt]]
      cur_y_list = [0] * 6
      cur_y_list[label] = 1
      posteriors = [float(x) for x in item[1:][1:-1]]
      try:
        priors = nid2prob[utt2nid[utt]]
      except KeyError:
        priors = [0]*6
      X.extend(zip(priors, posteriors))
      y.extend(cur_y_list)

  print(len(X), len(y))
  return X, y

def func(w, p1, p2):
  return np.abs(w[0])*p1 + np.abs(w[1])*p2

def residuals(w, y, p1, p2):
  return y - func(w, p1, p2)

def func2(w, p1, p2):
  if len(p1) == len(p2):
    p = []
    for prior, posterior in zip(p1, p2):
      p.append(np.abs(w) * prior + np.abs(([1]*6-w)) * posterior)
    return p
  return p2

def residuals2(w, y, p1, p2):
  result = 0
  p = func2(w, p1, p2)
  for y_, p_ in zip(y, p):
    logit = p_[y_]
    result -= math.log(logit)
  return result


# if summ < num_sum, use posteriors directly
def inference(w, test_x, test_y, priors, posteriors):
  # if len(test_x) == len(test_y) == len(priors) == len(posteriors):
  #   for i in range(len(test_x)):
  return False

if __name__ == '__main__':
  # nid2prob()
  # X, y = make_features('results/exp_36/posteriors_train3907_100', 10)
  # test_X, test_y = make_features('results/exp_36/posteriors_test300_100', 10, False)
  #
  # # initialize the weights
  # weights = np.array([0.2, 0.8])
  # priors = [x[0] for x in X]
  # posteriors = [x[1] for x in X]
  #
  # plsq = leastsq(residuals, weights, args=(np.array(y), np.array(priors), np.array(posteriors)))
  #
  # print(plsq[0])
  # print(plsq[0]/sum(plsq[0]))

  # X_priors, X_posteriors, y = read_priors_posteriors('results/exp_36/posteriors_train3907_100', 10)
  # test_X_priors, test_X_posteriors, test_y = read_priors_posteriors('results/exp_36/posteriors_test300_100', 10)
  #
  # # initialize the weights
  # weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
  # # print(len(func2(weights, np.array(X_priors), np.array(X_posteriors))))
  # plsq = minimize(residuals2, weights, args=(np.array(y), np.array(X_priors), np.array(X_posteriors)))
  #
  # print(plsq.x)
  # print(plsq.x/sum(plsq.x)) #  [0.20532031, 0.18516648, 0.14374873, 0.11432943, 0.21087999, 0.14055506]

  priors, posteriors, y = read_priors_posteriors('results/exp_36/posteriors_test300_100', 10)
  print(fixed_weight(0.4, priors, posteriors, y))
  print(fixed_weight([0.20532031, 0.18516648, 0.14374873, 0.11432943, 0.21087999, 0.14055506], priors, posteriors, y))
'''
{{0: {'recall': 0.9528, 'precision': 0.8067, 'f': 0.8736},
1: {'recall': 0.9153, 'precision': 0.7013, 'f': 0.7941},
2: {'recall': 0.0, 'precision': 0.0, 'f': 0.0},
3: {'recall': 0.2619, 'precision': 0.3929, 'f': 0.3143}, 
4: {'recall': 0.4286, 'precision': 0.3846, 'f': 0.4054},
5: {'recall': 0.0714, 'precision': 0.2, 'f': 0.1053},
'average_f': 0.4155, 'weighted_f': 0.6222, 'accuracy': 0.6733}
'''
