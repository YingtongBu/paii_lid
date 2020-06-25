import sklearn.metrics
import matplotlib.pyplot as plt
'''
nnm+nnh as positive
precision = TP / TP + FP
recall = TP / TP + FN
TP + FP: SUM(utt2lang_pred)
TP + FN: SUM(utt2lang_gold)
'''

def pr_value(boundary):
  utt2lang_pred = {}
  with open(f"posteriors_0610data_relabel", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nn = float(item[3])
      if nn > boundary:
        utt2lang_pred[utt] = [1, nn]
      else:
        utt2lang_pred[utt] = [0, nn]

  del utt2lang_pred['EE200609192556832-AN200609197569615']
  del utt2lang_pred['EE200609232902930-AN200610006152172']
  utt2lang_gold = {}
  TP = 0
  TN = 0
  with open(f"utt2lang_0610data_relabel_itg", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if utt not in utt2lang_pred.keys():
        continue
      if lang == 'nnl':
        del utt2lang_pred[utt]
        continue
      if lang in ['nnm', 'nnh', 'nn']:
        utt2lang_gold[utt] = 1
        if utt2lang_pred[utt][0] == 1:
          TP += 1
      else:
        utt2lang_gold[utt] = 0
        if utt2lang_pred[utt][0] == 0:
          TN += 1

  precision = TP / sum([x[1][0] for x in list(utt2lang_pred.items())])
  recall = TP / sum([x[1] for x in list(utt2lang_gold.items())])
  FPR = TN / sum([x[1]==0 for x in list(utt2lang_gold.items())])
  y_true = [utt2lang_gold[utt] for utt in utt2lang_pred.keys()]
  y_score = [x[1][1] for x in list(utt2lang_pred.items())]
  print(f"recall: {TP}, {sum([x[1][0] for x in list(utt2lang_pred.items())])}")
  print(f"precision: {precision}")
  print(f"recall: {recall}")
  print(f"FPR: {FPR}")

  return y_true, y_score, precision, recall, FPR


if __name__ == '__main__':
  '''calculate auc'''
  boundary = 0.5
  y_true, y_score, precision, recall, s = pr_value(boundary)
  print('auc:', sklearn.metrics.roc_auc_score(y_true, y_score))


  '''pr curve'''

  precision7 = []
  recall7 = []
  specificity7 = []
  exp = 'exp_23'
  for boundary in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95,
                   0.96, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999]:
    _, _, p, r, s = pr_value(boundary)
    precision7.append(p)
    recall7.append(r)
    specificity7.append(1-s)

  plt.figure()
  plt.plot(recall7, precision7, "b", linewidth=1)
  plt.xlabel("recall")
  plt.ylabel("precision")
  plt.xlim(0.0, 1.0)
  plt.ylim(0.0, 1.0)
  # plt.title("Line plot")
  plt.show()

  print(y_true)
  print(y_score)

