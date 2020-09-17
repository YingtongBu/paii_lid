import os
import json
import random
import pandas as pd
from collections import defaultdict, OrderedDict
from pa_nlp import nlp


def nationality_analysis(train_data, test_data):
  cols = ['total', 'us', 'uk', 'neutral', 'nn light', 'nn moderate', 'nn heavy']
  train = list(nlp.pydict_file_read(train_data))
  test = list(nlp.pydict_file_read(test_data))
  total_data = train + test
  nationality_dict = defaultdict()
  category_dict = defaultdict()
  data_list = []
  for d in total_data:
    if 'nationality' not in d:
      continue
    if d['nationality'] not in nationality_dict:
      nationality_dict[d.get('nationality')] = dict(total=0, category=[0] * 6)
    nationality_dict[d['nationality']]['total'] += 1
    nationality_dict[d['nationality']]['category'][d['class']] += 1
  for k, v in nationality_dict.items():
    print(k, v['total'], v['category'])
    data_list.append(v['total'])
    data_len = len(data_list)
    for c in v['category']:
      data_list[data_len-1].append(c)



  # for k, v in nationality_dict.items():
  #   for c in v:
  #



if __name__ == '__main__':
    train_data = 'data/data_til0819/data.03.train.pydict'
    test_data = 'data/data_til0819/data.03.test.pydict'
    nationality_analysis(train_data, test_data)