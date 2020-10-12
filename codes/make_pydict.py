import os

def pydict(ivector_paths, outname, xvector_paths=None):
  utt2nid = {}
  utt2lang = {}
  # nid2prob = {}
  # with open('data/nid2prob.v5', 'r') as f:
  #   for line in f:
  #     item = line.strip().split()
  #     nid = item[0]
  #     summ = int(item[1])
  #     nid2prob[nid] = [summ]
  #     for i in range(2, 8):
  #       nid2prob[nid].append(float(item[i]))
  with open('data/utt2nid_train', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nid = item[-1]
      utt2nid[utt] = nid
  with open('data_train_itg_0903/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if lang == 'Others':
        continue
      utt2lang[utt] = lang
  label_dict = {'us': 0, 'uk': 1, 'nnn': 2, 'nnl': 3, 'nnm': 4, 'nnh': 5}
  content = {}
  for ivector_path in ivector_paths:
    files = os.listdir(ivector_path)
    for file in files:
      with open(f'{ivector_path}/{file}', 'r') as f:
        for line in f:
          item = line.strip().split()
          utt = item[0]
          label = utt2lang[utt]
          vector = [float(x) for x in item[1:][1:-1]]
          length = len(vector)
          if utt in content:
            content[utt][f'ivector_{length}'] = vector
          else:
            try:
              content[utt] = {}
              content[utt]['id'] = utt
              content[utt]['class'] = label_dict[label]
              content[utt]['nationality'] = utt2nid[utt]
              # content[utt]['nationality2prob'] = nid2prob[nid]
            except KeyError:
              continue
            content[utt][f'ivector_{length}'] = vector

  for xvector_path in xvector_paths:
    files = os.listdir(xvector_path)
    for file in files:
      with open(f'{xvector_path}/{file}', 'r') as f:
        for line in f:
          item = line.strip().split()
          utt_aug = item[0]
          # label = utt_aug.split('-')[0]
          utt = utt_aug.split('-')[1].replace('.', '-')
          label = utt2lang[utt]
          vector = [float(x) for x in item[1:][1:-1]]
          length = len(vector)
          if utt in content:
            content[utt][f'xvector_{length}'] = vector
          else:
            try:
              content[utt] = {}
              content[utt]['id'] = utt
              content[utt]['class'] = label_dict[label]
              content[utt]['nationality'] = utt2nid[utt]
              # content[utt]['nationality2prob'] = nid2prob[nid]
            except KeyError:
              continue
            content[utt][f'xvector_{length}'] = vector

  print(len(list(content.values())))
  print(list(content.values())[2].keys())
  with open(outname, 'w') as f:
    f.writelines(str(line) + '\n' for line in list(content.values()))

def pydict2(ivector_paths, utt_path_list, outname, xvector_paths=None):
  utt2nid = {}
  utt2lang = {}
  utt2path = {}
  utt2align = {}
  for dataset in ['train', 'test']:
    with open(f'pydict/tmp/data.03.{dataset}.pydict.v2', 'r') as f:
      for line in f:
        line = eval(line)
        utt = line['id']
        utt2lang[utt] = line['class']
        utt2nid[utt] = line['nationality']
        utt2path[utt] = line['path']
        utt2align[utt] = line['align_res']

  assert len(utt2lang) == len(utt2nid) == len(utt2path) == len(utt2align)
  print(len(utt2lang))

  utt_list = []
  with open(utt_path_list, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      utt_list.append(utt)

  content = {}
  for utt in utt_list:
    try:
      content[utt] = {}
      content[utt]['id'] = utt
      content[utt]['class'] = utt2lang[utt]
      content[utt]['nationality'] = utt2nid[utt]
      content[utt]['path'] = utt2path[utt]
      content[utt]['align_res'] = utt2align[utt]
      # content[utt]['pitch_seq'] = cal_pitch(utt2path[utt])
    except KeyError:
      continue

  print(len(content))

  for ivector_path in ivector_paths:
    files = os.listdir(ivector_path)
    for file in files:
      with open(f'{ivector_path}/{file}', 'r') as f:
        for line in f:
          item = line.strip().split()
          utt = item[0]
          vector = [float(x) for x in item[1:][1:-1]]
          length = len(vector)
          content[utt][f'ivector_{length}'] = vector

  for xvector_path in xvector_paths:
    files = os.listdir(xvector_path)
    for file in files:
      with open(f'{xvector_path}/{file}', 'r') as f:
        for line in f:
          item = line.strip().split()
          utt_aug = item[0]
          vector = [float(x) for x in item[1:][1:-1]]
          length = len(vector)
          if 'plp' in xvector_path:
            # utt = utt_aug
            utt = f"{utt_aug.split('-')[1]}-{utt_aug.split('-')[2]}"
            content[utt][f'xvector_plp_{length}'] = vector
          else:
            utt = utt_aug.split('-')[1].replace('.', '-')
            content[utt][f'xvector_{length}'] = vector

  print(len(list(content.values())))
  print(list(content.values())[2].keys())
  with open(outname, 'w') as f:
    f.writelines(str(line) + '\n' for line in list(content.values()))


def correct_label():
  utt2lang = {}
  utt2nid = {}
  utt2nationality2prob = {}
  utt2ivector100 = {}
  utt2ivector200 = {}
  utt2ivector600 = {}
  utt2xvector512 = {}

  with open('pydict/data.00.train.pydict', 'r') as f:
    for line in f:
      line = eval(line)
      utt = line['id']
      utt2lang[utt] = line['class']
      utt2nid[utt] = line['nationality']
      utt2nationality2prob[utt] = line['nationality2prob']
      utt2ivector100[utt] = line['ivector_100']
      utt2ivector200[utt] = line['ivector_200']
      utt2ivector600[utt] = line['ivector_600']
      utt2xvector512[utt] = line['xvector_512']
  assert len(utt2lang) == len(utt2nid) == len(utt2nationality2prob) \
         == len(utt2ivector100) == len(utt2ivector200) == len(utt2ivector600) \
         == len(utt2xvector512)
  print(len(utt2lang))

  pydict_content = []
  with open('/Users/buyingtong/User/Paii/nlp/repos/nlp_team/speech/accent_dection/data/data.02.train.pydict', 'r') as f:
    for line in f:
      line = eval(line)
      cur = {}
      utt = line['id']
      label = utt2lang[utt]
      align_res = line['align_res']
      cur['id'] = utt
      cur['class'] = label
      cur['nationality'] = utt2nid[utt]
      cur['nationality2prob'] = utt2nationality2prob[utt]
      cur['align_res'] = align_res
      cur['ivector_100'] = utt2ivector100[utt]
      cur['ivector_200'] = utt2ivector200[utt]
      cur['ivector_600'] = utt2ivector600[utt]
      cur['xvector_512'] = utt2xvector512[utt]

      pydict_content.append(cur)
  print(len(pydict_content))
  print(pydict_content[2].keys())
  with open('pydict/data.00.train.pydict.v2', 'w') as f:
    f.writelines(str(line) + '\n' for line in pydict_content)

def data_four_nn():
  # utt2nid = {}
  # utt2path = {}
  # with open('data/utt2nid', 'r') as f:
  #   for line in f:
  #     item = line.split()
  #     utt = item[0]
  #     nid = item[-1]
  #     utt2nid[utt] = nid
  #
  # # with open('data/data_mturk_train6547/wav.scp_mturk_6547', 'r') as f:
  # with open('data/test_mturk_300/wav.scp', 'r') as f:
  #   for line in f:
  #     item = line.split()
  #     utt = item[0]
  #     path = item[-1]
  #     recording = path.split('/')[-1]
  #     date = path.split('/')[-2]
  #     new_path = f'/data/workspace/buyingtong/teacher_accent_classification/wav/{date}/{recording}'
  #     utt2path[utt] = new_path

  pydict_content = []
  with open('data.03.test.pydict', 'r') as f:
    for line in f:
      line = eval(line)
      try:
        label = line['class']
      except KeyError:
        print(line)
        continue
      if label in [0, 1]:
        continue
      cur = {}
      utt = line['id']
      cur['id'] = utt
      cur['class'] = label
      cur['nationality'] = line['nationality']
      cur['path'] = line['path']
      cur['align_res'] = line['align_res']
      cur['pitch_seq'] = line['pitch_seq']
      try:
        cur['ivector_100'] = line['ivector_100']
      except KeyError:
        cur['ivector_100'] = []
      try:
        cur['ivector_200'] = line['ivector_200']
      except KeyError:
        cur['ivector_200'] = []
      try:
        cur['ivector_600'] = line['ivector_600']
      except KeyError:
        cur['ivector_600'] = []
      try:
        cur['xvector_200'] = line['xvector_200']
      except KeyError:
        cur['xvector_200'] = []
      try:
        cur['xvector_512'] = line['xvector_512']
      except KeyError:
        cur['xvector_512'] = []
      try:
        cur['xvector_plp_512'] = line['xvector_plp_512']
      except KeyError:
        cur['xvector_plp_512'] = []
      pydict_content.append(cur)
  print(len(pydict_content))
  print(pydict_content[2].keys())
  with open('data.03.four-unstandard-accents.test.pydict', 'w') as f:
    f.writelines(str(line) + '\n' for line in pydict_content)

def select_testset_data03():
  to_write = []
  train = []
  us = 0
  uk = 0
  nnl = 0
  nnn = 0
  nnm = 0
  nnh = 0
  with open('mturk/utt2lang6.v9', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      if lang == 'Others':
        continue
      if lang == 'us' and us < 436:
        us += 1
        to_write.append(utt + ' ' + lang + '\n')
      elif lang == 'uk' and uk < 247:
        uk += 1
        to_write.append(utt + ' ' + lang + '\n')
      elif lang == 'nnl' and nnl < 267:
        nnl += 1
        to_write.append(utt + ' ' + lang + '\n')
      elif lang == 'nnm' and nnm < 208:
        nnm += 1
        to_write.append(utt + ' ' + lang + '\n')
      elif lang == 'nnn' and nnn < 137:
        nnn += 1
        to_write.append(utt + ' ' + lang + '\n')
      elif lang == 'nnh' and nnh < 73:
        nnh += 1
        to_write.append(utt + ' ' + lang + '\n')
      else:
        train.append(utt + ' ' + lang + '\n')
  print(len(to_write), len(train))
  with open(f'data/utt2lang_mturk_test1368', 'w') as f:
    f.writelines(line for line in to_write)
  with open(f'data/utt2lang_mturk_train5479', 'w') as f:
    f.writelines(line for line in train)

def filter_align_fail():
  utt2wav = []
  with open('data.03.train.pydict', 'r') as f:
    for line in f:
      line = eval(line)
      utt = line['id']
      try:
        path = line['path']
      except KeyError:
        print('no path', utt)
        continue
      align = line['align_res']
      if align:
        utt2wav.append(utt + ' ' + path + '\n')
  print(len(utt2wav))
  with open(f'wav.scp.04.train', 'w') as f:
    f.writelines(line for line in utt2wav)

def filter_align_fail_pydict():
  pydict_content = []
  with open('data.03.four-unstandard-accents.train.pydict', 'r') as f:
    for line in f:
      line = eval(line)
      try:
        label = line['class']
      except KeyError:
        print(line)
        continue
      align = line['align_res']
      print(isinstance(align, list))
      if align == []:
        print('align fail:', line['id'])
        continue
      cur = {}
      utt = line['id']
      cur['id'] = utt
      cur['class'] = label
      cur['nationality'] = line['nationality']
      cur['path'] = line['path']
      cur['align_res'] = line['align_res']
      cur['pitch_seq'] = line['pitch_seq']
      try:
        cur['ivector_100'] = line['ivector_100']
      except KeyError:
        cur['ivector_100'] = []
      try:
        cur['ivector_200'] = line['ivector_200']
      except KeyError:
        cur['ivector_200'] = []
      try:
        cur['ivector_600'] = line['ivector_600']
      except KeyError:
        cur['ivector_600'] = []
      try:
        cur['xvector_200'] = line['xvector_200']
      except KeyError:
        cur['xvector_200'] = []
      try:
        cur['xvector_512'] = line['xvector_512']
      except KeyError:
        cur['xvector_512'] = []
      try:
        cur['xvector_plp_512'] = line['xvector_plp_512']
      except KeyError:
        cur['xvector_plp_512'] = []
      pydict_content.append(cur)
  print(len(pydict_content))
  print(pydict_content[2].keys())
  with open('data.04.non-std-accent.train.pydict', 'w') as f:
    f.writelines(str(line) + '\n' for line in pydict_content)



import librosa
def cal_pitch(audio_path='/data/pytong/wav/itg_0603/EE1811306983070-interview-iOS-2020-05-29-09-06-23.wav',
              sample_rate=16000, hop=160):
  signal, rate = librosa.load(audio_path, sr=sample_rate, mono=True)
  pitches, magnitudes = librosa.piptrack(y=signal, sr=rate,
                                         hop_length=hop)
  pitch_result = []
  pitches_reverted = list(zip(*pitches))
  magnitudes_reverted = list(zip(*magnitudes))
  for cur_pitch_list, cur_magnitude_list in zip(pitches_reverted, magnitudes_reverted):
    index = cur_magnitude_list.index(max(cur_magnitude_list))
    pitch = cur_pitch_list[index]
    pitch_result.append(pitch)
  print(len(pitch_result), pitch_result[0:2])
  return pitch_result

def pydict3(ivector_paths, utt_path_list, outname, xvector_paths=None):
  utt2nid = {}
  utt2lang = {}
  utt2path = {}
  utt2align = {}
  for dataset in ['train', 'test']:
    with open(f'pydict/tmp/data.03.{dataset}.pydict.v2', 'r') as f:
      for line in f:
        line = eval(line)
        utt = line['id']
        utt2lang[utt] = line['class']
        utt2nid[utt] = line['nationality']
        utt2path[utt] = line['path']
        # utt2align[utt] = line['align_res']

  assert len(utt2lang) == len(utt2nid) == len(utt2path) == len(utt2align)
  print(len(utt2lang))

  utt_list = []
  with open(utt_path_list, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      utt_list.append(utt)

  content = {}
  for utt in utt_list:
    try:
      content[utt] = {}
      content[utt]['id'] = utt
      content[utt]['class'] = utt2lang[utt]
      content[utt]['nationality'] = utt2nid[utt]
      content[utt]['path'] = utt2path[utt]
      # content[utt]['align_res'] = utt2align[utt]
      # content[utt]['pitch_seq'] = cal_pitch(utt2path[utt])
    except KeyError:
      continue

  print(len(content))

  for ivector_path in ivector_paths:
    files = os.listdir(ivector_path)
    for file in files:
      with open(f'{ivector_path}/{file}', 'r') as f:
        for line in f:
          item = line.strip().split()
          utt = item[0]
          vector = [float(x) for x in item[1:][1:-1]]
          length = len(vector)
          content[utt][f'ivector_{length}'] = vector

  for xvector_path in xvector_paths:
    files = os.listdir(xvector_path)
    for file in files:
      with open(f'{xvector_path}/{file}', 'r') as f:
        for line in f:
          item = line.strip().split()
          utt_aug = item[0]
          vector = [float(x) for x in item[1:][1:-1]]
          length = len(vector)
          if 'plp' in xvector_path:
            # utt = utt_aug
            utt = f"{utt_aug.split('-')[1]}-{utt_aug.split('-')[2]}"
            content[utt][f'xvector_plp_{length}'] = vector
          else:
            utt = utt_aug.split('-')[1].replace('.', '-')
            content[utt][f'xvector_{length}'] = vector

  print(len(list(content.values())))
  print(list(content.values())[2].keys())
  with open(outname, 'w') as f:
    f.writelines(str(line) + '\n' for line in list(content.values()))






if __name__ == '__main__':
  # pydict2(['results/data.03.related/ivectors.test_mturk_1368_100',
  #          'results/data.03.related/ivectors.test_mturk_1368_200',
  #          'results/data.03.related/ivectors.test_mturk_1368_600'],
  #         'data/test_mturk_1368/wav.scp', 'pydict/data.03.test.pydict',
  #         ['results/data.03.related/xvectors.test_mturk_1368_200',
  #          'results/data.03.related/xvectors.test_mturk_1368_512',
  #          'results/data.03.related/xvectors.test_mturk_1368_plp_512'])
  # pydict2(['ivectors.data_mturk_train5479_100',
  #          'ivectors.data_mturk_train5479_200',
  #          'ivectors.data_mturk_train5479_600'],
  #         'wav.scp.train5479', 'data.03.train.pydict',
  #         ['xvectors.data_mturk_train5479_200',
  #          'xvectors.data_mturk_train5479_512',
  #          'xvectors.data_mturk_train5479_plp_512'])
  # cal_pitch()
  # filter_align_fail()
  # select_testset_data03()
  # data_four_nn()
  # correct_label()
  pydict(['all_itg_data/ivectors.data_train_itg_0903_100',
          'all_itg_data/ivectors.data_train_itg_0903_200',
          'all_itg_data/ivectors.data_train_itg_0903_600'],
         'data.06.train.pydict',
         ['all_itg_data/xvectors.data_train_itg_0903_xvector_512_2'])
  # pydict(['results/exp_36/ivectors.test_mturk_300_100',
  #         'results/exp_36/ivectors.test_mturk_300_200',
  #         'results/exp_36/ivectors.test_mturk_300'],
  #        'pydict/data.00.test.pydict',
  #        ['results/xvectors_test.4'])
