import json
import requests
import os
import numpy as np

# 'curl -X POST -F file=@audio.wav -F transcript=@trans.txt 192.168.1.209:9001/alig'
#
# res = requests.post('http://192.168.1.209:9001/alig',
#                     files={'file': open('EE200617030838195-interview-44Cpa-2020-06-16T19-36-46-407Z.wav', 'rb'), 'transcript': open('trans.txt', 'rb')}).json()
# # results = json.dump(res, open(res, "w"))
# # results = json.loads(res)
# results_list = res['alignment']
# words = [r[0] for r in results_list]
# print(words)
#
# with open('test.json', 'r') as f:
#   for line in f:
#     results = json.loads(line)
#     results_list = results['alignment']
#     words = [r[0] for r in results_list]
#     print(words)


def add_train():
  utt2path = {}
  with open('data/data_til0819/wav.scp', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      path = item[-1]
      utt2path[utt] = path
  test_utt = []
  with open('data/test_mturk_300/utt2lang', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      test_utt.append(utt)
  utt2path_train = []
  utt2lang_train = []
  with open('mturk/utt2lang6.v9', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      if utt in test_utt:
        continue
      lang = item[-1]
      if lang == 'Others':
        continue
      path = utt2path[utt]
      utt2path_train.append(utt + ' ' + path + '\n')
      utt2lang_train.append(utt + ' ' + lang + '\n')
  print(len(utt2lang_train))
  with open('data/utt2lang_mturk_6547', 'w') as f:
    f.writelines(line for line in utt2lang_train)
  with open('data/wav.scp_mturk_6547', 'w') as f:
    f.writelines(line for line in utt2path_train)

def run_align(wavs, utt2lang_file, out_name):
  tag2id = {'us': 0, 'uk': 1, 'nnn': 2, 'nnl': 3, 'nnm': 4, 'nnh': 5}
  already_utt = []
  utt2align = {}
  with open('data.04.train.pydict', 'r') as f:
    for line in f:
      line = eval(line)
      utt = line['id']
      already_utt.append(utt)
      utt2align[utt] = line['align_res']
  utt2lang = {}
  with open(utt2lang_file, 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
  pydict_content = []
  with open(wavs, 'r') as f:
    for line in f:
      item = line.split()
      cur = {}
      utt = item[0]
      lang = utt2lang[utt]
      try:
        lang_id = tag2id[lang]
      except KeyError:
        continue
      if utt in already_utt:
        print('already')
        cur['id'] = utt
        cur['class'] = lang_id
        cur['align_res'] = utt2align[utt]
      else:
        print('new')
        path = item[-1]
        try:
          res = requests.post('http://192.168.1.209:9001/alig',
                              files={'file': open(path, 'rb'),
                                     'transcript': open('trans.txt', 'rb')}
                              ).json()['alignment']
        except FileNotFoundError:
          continue
        res_filter_eps = [r for r in res if r[0] != '<eps>']
        cur['id'] = utt
        cur['class'] = lang_id
        cur['align_res'] = res_filter_eps
      pydict_content.append(cur)
  print(len(pydict_content))
  with open(out_name, 'w') as f:
    f.writelines(str(line) + '\n' for line in pydict_content)

# run_align('data_mturk_train6255/wav.scp_mturk_6255', 'data_mturk_train6255/utt2lang_mturk_6255', 'data.03.train.pydict')
# run_align('data_mturk_train6547/wav.scp_mturk_6547', 'data_mturk_train6547/utt2lang_mturk_6547', 'data.03.train.pydict')

def run_align2(data_fn, output_fn):
  pydict_content = []
  with open(data_fn, 'r') as f:
    for line in f:
      cur = {}
      line = eval(line)
      wav_path = line['LBD']
      trans  = wav_path.replace('wav', 'txt')
      try:
        res = requests.post('http://192.168.1.209:9001/alig',
                            files={'file': open(wav_path, 'rb'),
                                   'transcript': open(trans, 'rb')}
                            ).json()['alignment']
      except FileNotFoundError:
        continue
      res_filter_eps = [r for r in res if r[0] != '<eps>']
      cur['meta'] = line['meta']
      cur['label'] = line['label']
      cur['align_res'] = res_filter_eps
      pydict_content.append(cur)
  print(len(pydict_content))
  with open(output_fn, 'w') as f_o:
    f_o.writelines(str(line) + '\n' for line in pydict_content)

if __name__ == '__main__':
  add_train()