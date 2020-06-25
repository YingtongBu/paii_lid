import sklearn.metrics
import matplotlib.pyplot as plt

'''
nn as positive
precision = TP / TP + FP
recall = TP / TP + FN
TP + FP: SUM(utt2lang_pred)
TP + FN: SUM(utt2lang_gold)
'''

# use binary classification result to get three classes
def binary_wer(posteriors, gold, boundary):
  utt2lang_pred = {}
  with open(posteriors, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      us = float(item[2])
      if us >= 0.5 + boundary:
        utt2lang_pred[utt] = 'us'
      if 0.5 - boundary < us < 0.5 + boundary:
        utt2lang_pred[utt] = 'nn'
      else:
        utt2lang_pred[utt] = 'uk'
  # correct = 0
  TP, P = 0, 0
  with open(gold, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      if lang == 'nn':
        print(utt2lang_pred[utt])
        P += 1
        if utt2lang_pred[utt] == 'nn':
          TP += 1
  return TP/P
  #     if lang == utt2lang_pred[utt]:
  #       correct += 1
  # return correct/len(utt2lang_pred)


def nn_wer(gold_file, pred_file):
  pred = {}
  P = 0
  TP = 0
  with open(pred_file, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      pred[utt] = lang
  with open(gold_file, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      if lang == 'nn':
        P += 1
        if pred[utt] == 'nn':
          TP += 1
  ER = 1-TP/P
  return ER

def pr_value(exp, boundary):
  utt2lang_pred = {}
  with open(f"results/v1/{exp}/posteriors", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nn = float(item[4])
      if nn > boundary:
        utt2lang_pred[utt] = [1, nn]
      else:
        utt2lang_pred[utt] = [0, nn]

  utt2lang_gold = {}
  TP = 0
  with open(f"data/lre07_all3/utt2lang", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if utt not in utt2lang_pred.keys():
        continue
      if lang == 'nn':
        utt2lang_gold[utt] = 1
        if utt2lang_pred[utt][0] == 1:
          TP += 1
      else:
        utt2lang_gold[utt] = 0

  precision = TP/sum([x[1][0] for x in list(utt2lang_pred.items())])
  recall = TP/sum([x[1] for x in list(utt2lang_gold.items())])
  y_true = [utt2lang_gold[x] for x in utt2lang_pred.keys()]
  y_score = [x[1][1] for x in list(utt2lang_pred.items())]
  print(f"precision: {precision}")
  print(f"recall: {recall}")

  return y_true, y_score, precision, recall

def pr_value2(boundary):
  pred = []
  nnscore = []
  with open(f"results/v1/exp_7/gaussian_classifier/posteriors", 'r') as f:
    for line in f:
      item = line.strip().split()
      nn = item[-1][:-1]
      nn = float(nn)
      if nn > boundary:
        pred.append(1)
      else:
        pred.append(0)
      nnscore.append(nn)

  gold = []
  with open(f"results/v1/exp_7/gaussian_classifier/gold", 'r') as f:
    for line in f:
      item = line.strip()
      if item == '2':
        gold.append(1)
      else:
        gold.append(0)

  TP = sum([x==y==1 for x, y in zip(pred, gold)])
  precision = TP/sum(pred)
  recall = TP/sum(gold)
  y_true = gold
  y_score = nnscore
  print(f"precision: {precision}")
  print(f"recall: {recall}")

  return y_true, y_score, precision, recall

y_true, y_score, precision, recall = pr_value2(0.2)
print(sklearn.metrics.roc_auc_score(y_true, y_score))

'''calculate auc'''
exp = 'exp_7'
boundary = 0.7
y_true, y_score, precision, recall = pr_value(exp, boundary)
print(sklearn.metrics.roc_auc_score(y_true, y_score))


'''pr curve'''
# precision6 = [0]
# recall6 = [1]
# exp = 'exp_6'
# for boundary in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95,
#                  0.96, 0.97, 0.98, 0.99]:
#   p, r = pr_value(exp, boundary)
#   precision6.append(p)
#   recall6.append(r)
#
# precision6.append(1)
# recall6.append(0)
#
# precision7 = [0]
# recall7 = [1]
precision7 = []
recall7 = []
exp = 'exp_7'
for boundary in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.99]:
  _, _, p, r = pr_value(exp, boundary)
  precision7.append(p)
  recall7.append(r)

# precision7.append(1)
# recall7.append(0)

'''
plt.figure()
# plt.plot(recall6, precision6, "b", linewidth=1)
plt.plot(recall7, precision7, "b", linewidth=1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
# plt.title("Line plot")
plt.show()
'''