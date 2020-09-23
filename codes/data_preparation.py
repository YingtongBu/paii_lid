#!/usr/bin/env python
#coding: utf-8
#author: Xinlu Yu(xinlu.yu1@pactera.com)

import json
import os, sys
import subprocess

map6 = {
  'us':0,
  'uk':1,
  'nnn':2,
  'nnl':3,
  'nnm':4,
  'nnh':5
}


map2 = {
  0:0,
  1:0,
  2:0,
  3:0,
  4:1,
  5: 1
  }
def download_data(data_json):
  cnt = 0
  data = json.load(open(data_json, 'r'))
  for k, v in data.items():
    for wav in v.get('wav_path'):
      result_dir = f'/Users/yuxinlu/pa_work/paii_lid/mturk/sample_500/{k}'
      if not os.path.exists(result_dir):
        os.makedirs(result_dir)
      cmd = f'scp -P 39002 -r lulu@ml.pingan-labs.us:{wav} {result_dir}'
      os.system(cmd)
      wav_base = wav.split('/')[-1]
      print(f'successful download {wav_base}to {result_dir}')
      cnt +=1
      print(cnt)

def gen_itg_label(input_data, itg_label_file, output_fn):
  itg_label =  dict()
  itg_results = []
  itg_label_data = [item.strip().split()
                    for item in open(itg_label_file).readlines()]

  for d in itg_label_data:
    if d[0] not in itg_label:
      itg_label[d[0]] = d[1]
  count = 0
  with open(input_data, 'r') as f:
    for line in f:
      cur = {}
      line = eval(line)
      line['class'] = map6[itg_label[line['id']]]
      # print(line.keys())
      count +=1
      print(count)
      itg_results.append(line)
  print(len(itg_results))
  with open(output_fn, 'w') as f_o:
    f_o.writelines(str(line) + '\n' for line in itg_results)


if __name__ == '__main__':
    # data = 'mturk/sample_500/sample_5.json'
    # download_data(data)
    input_data = 'data/data_til0819/data.03.test.pydict'
    itg_label_file = 'data/data_til0819/utt2lang'
    output_file = 'data/data_til0819/data.05.test.pydict'
    gen_itg_label(input_data, itg_label_file, output_file)
