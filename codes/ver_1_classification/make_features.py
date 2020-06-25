
import os

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


def read_ivector(ivector_path, data_path):
  utt2lang = {}
  for dataset in ['train', 'lre07']:
    with open(f'data/{dataset}_{data_path}/utt2lang', 'r') as f:
      for line in f:
        item = line.strip().split()
        utt = item[0]
        lang = item[-1]
        utt2lang[utt] = lang
  X = []
  y = []
  files = os.listdir(ivector_path)
  for file in files:
    with open(f'{ivector_path}/{file}', 'r') as f:
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