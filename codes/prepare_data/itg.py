
import csv
import os
import random
import numpy as np
import sys
import xlrd


def listdir_nohidden(path):
  res = [f for f in os.listdir(path) if not f.startswith('.')]
  return res

def download():
  utt = []
  with open('teacher_accent_4000utt.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
      link = row[-2]
      utt_name = link.split('/')[-1]
      if utt_name in os.listdir('/data/pytong/wav/itg_4000'):
        utt.append(utt_name)
        continue
      os.system(f'wget {row[-2]}')

def get_nation():
  utt2nation = []
  nations = []
  with open('teacher_accent_4000utt.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    index = 0
    for row in reader:
      index = index + 1
      spk = row[0]
      nation = row[2]
      utt = f'{spk}-itg{index:04d}'
      utt2nation.append(utt + ' ' + nation + '\n')
      if nation.lower() not in nations:
        nations.append(nation.lower())
  # with open(f'data/utt2nation', 'w') as f:
  #   f.writelines(line for line in utt2nation)

  return nations

def utt2lang():
  utt2lang = [] # lang in [us, uk, africa, india, hongkong, philippines, singapore, australia, newzealand, china, othernn]
  with open('teacher_accent_4000utt.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    index = 0
    for row in reader:
      index = index + 1
      spk = row[0]
      utt = f'{spk}-itg{index:04d}'
      nation = row[2].lower().replace(' ', '')
      print(nation)
      tag = row[-1]
      if tag == 'Native-American':
        lang = 'us'
      elif tag == 'Native-British':
        lang = 'uk'
      else:
        if nation == 'southafrica':
          lang = 'africa'
        elif nation == 'taiwan,provinceofchina':
          lang = 'china'
        if nation in ['africa', 'india', 'philippines', 'singapore',
                      'australia', 'newzealand', 'china']:
          lang = nation
        else:
          lang = 'nn'

      utt2lang.append(utt + ' ' + lang + '\n')

  print('total:', len(utt2lang))
  random.seed(0)
  lre07_indices = sorted(random.sample(range(len(utt2lang)), int((len(utt2lang))/10)))
  print(lre07_indices)
  train_indices = list(range(len(utt2lang)))
  for index in lre07_indices:
    train_indices.remove(index)
  print(train_indices)

  for dataset in ['train', 'lre07']:

    with open(f'data/{dataset}_itg_filtered/utt2lang', 'w') as f:
      f.writelines(line for line in list(np.array(utt2lang)[locals()[f'{dataset}_indices']]))

def random_selection():
  us = []
  uk = []
  no_light = []
  no_neutral = []
  no_moderate = []
  no_heavy = []

  with open('teacher_accent_4000utt.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    index = 0
    for row in reader:
      index = index + 1
      link = row[-2]
      utt_name = link.split('/')[-1]
      tag = row[-1]
      if tag == 'Native-American':
        us.append([link, tag])
      elif tag == 'Native-British':
        uk.append([link, tag])
      elif tag == 'Non-native - Light':
        no_light.append([link, tag])
      elif tag == 'Non-native - Neutral':
        no_neutral.append([link, tag])
      elif tag == 'Non-native - Heavy':
        no_heavy.append([link, tag])
      elif tag == 'Non-native - Moderate':
        no_moderate.append([link, tag])

  random.seed(0)
  us_indices = sorted(random.sample(range(len(us)), 20))
  uk_indices = sorted(random.sample(range(len(uk)), 20))
  no_light_indices = sorted(random.sample(range(len(no_light)), 15))
  no_neutral_indices = sorted(random.sample(range(len(no_neutral)), 15))
  no_heavy_indices = sorted(random.sample(range(len(no_heavy)), 15))
  no_moderate_indices = sorted(random.sample(range(len(no_moderate)), 15))

  for tag in ['us', 'uk', 'no_moderate', 'no_light', 'no_neutral', 'no_heavy']:
    with open('random_selection.csv', 'a') as f:
      writer = csv.writer(f, delimiter=',')
      writer.writerows(line for line in list(np.array(locals()[f'{tag}'])[locals()[f'{tag}_indices']]))

def yingtong():
  gold = []
  pred = []
  with open('random_selection.2.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    index = 0
    for row in reader:
      index = index + 1
      itg = row[2]
      yingtong = row[5]

      gold.append(itg)
      if yingtong == '3':
        pred.append('Non-native - Light')
      elif yingtong == '4':
        pred.append('Non-native - Neutral')
      elif yingtong == '5':
        pred.append('Non-native - Heavy')
      elif yingtong == '6':
        pred.append('Non-native - Moderate')
      elif yingtong == '1':
        pred.append('Native-American')
      else:
        pred.append('Native-British')

      # if itg in ['Non-native - Light', 'Non-native - Neutral', 'Non-native - Heavy', 'Non-native - Moderate']:
      #   gold.append('nn')
      # else:
      #   gold.append(itg)
      # if yingtong in ['3', '4', '5', '6']:
      #   pred.append('nn')
      # elif yingtong == '1':
      #   pred.append('Native-American')
      # else:
      #   pred.append('Native-British')

      # if itg in ['Non-native - Light', 'Non-native - Neutral']:
      #   gold.append('nn-light')
      # elif itg in ['Non-native - Heavy', 'Non-native - Moderate']:
      #   gold.append('nn-heavy')
      # else:
      #   gold.append(itg)
      # if yingtong in ['3', '4']:
      #   pred.append('nn-light')
      # elif yingtong in ['5', '6']:
      #   pred.append('nn-heavy')
      # elif yingtong == '1':
      #   pred.append('Native-American')
      # else:
      #   pred.append('Native-British')

  print(len(gold), gold)
  print(len(pred), pred)
  print(sum([x == y for x, y in zip(gold, pred)]))

def itg_4000():
  utt2wav = []
  utt2lang = []
  utt2spk = []
  unique_spks = []
  us = 0
  uk = 0
  no_light = 0
  no_neutral = 0
  no_moderate = 0
  no_heavy = 0

  with open('teacher_accent_4000utt.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    index = 0
    for row in reader:
      index = index + 1
      spk = row[0]
      utt = f'{spk}-itg{index:04d}'
      link = row[-2]
      utt_name = link.split('/')[-1]
      dir = f'/data/pytong/wav/itg_4000/standard/{utt_name}'
      tag = row[-1]
      # if spk in unique_spks:
      #   continue
      # unique_spks.append(spk)
      # if tag == 'Native-American':
      #   us += 1
      # elif tag == 'Native-British':
      #   uk += 1
      # elif tag == 'Non-native - Light':
      #   no_light += 1
      # elif tag == 'Non-native - Neutral':
      #   no_neutral += 1
      # elif tag == 'Non-native - Heavy':
      #   no_heavy += 1
      # elif tag == 'Non-native - Moderate':
      #   no_moderate += 1

      # print(us, uk, no_light, no_neutral, no_moderate, no_heavy)
      if tag == 'Non-native - Light':
        lang = 'light'
      elif tag == 'Native-American':
        lang = 'us'
      elif tag == 'Native-British':
        lang = 'uk'
      else:
        lang = 'nn' # continue

      utt2lang.append(utt + ' ' + lang + '\n')
      utt2spk.append(utt + ' ' + spk + '\n')
      utt2wav.append(utt + ' ' + dir + '\n')

  print('total:', len(utt2spk))
  random.seed(0)
  lre07_indices = sorted(random.sample(range(len(utt2wav)), int((len(utt2wav))/10)))
  print(lre07_indices)
  train_indices = list(range(len(utt2wav)))
  for index in lre07_indices:
    train_indices.remove(index)
  print(train_indices)

  for dataset in ['train', 'lre07']:
    with open(f'data/{dataset}_itg3_v5/wav.scp1', 'w') as f:
      f.writelines(line for line in list(np.array(utt2wav)[locals()[f'{dataset}_indices']]))
    with open(f'data/{dataset}_itg3_v5/utt2lang1', 'w') as f:
      f.writelines(line for line in list(np.array(utt2lang)[locals()[f'{dataset}_indices']]))
    with open(f'data/{dataset}_itg3_v5/utt2spk1', 'w') as f:
      f.writelines(line for line in list(np.array(utt2spk)[locals()[f'{dataset}_indices']]))

def delete_light():
  for dataset in ['train', 'lre07']:
    utt_delete = []
    with open(f'data/{dataset}_itg3_v5/utt2lang1', 'r') as f:
      for line in f:
        item = line.strip().split()
        lang = item[-1]
        utt = item[0]
        if lang == 'light':
          utt_delete.append(utt)

    utt2wav = []
    utt2lang = []
    utt2spk = []

    with open(f'data/{dataset}_itg3_v5/utt2lang1', 'r') as f:
      for line in f:
        item = line.strip().split()
        lang = item[-1]
        utt = item[0]
        if utt not in utt_delete:
          utt2lang.append(utt + ' ' + lang + '\n')

    with open(f'data/{dataset}_itg3_v5/utt2spk1', 'r') as f:
      for line in f:
        item = line.strip().split()
        spk = item[-1]
        utt = item[0]
        if utt not in utt_delete:
          utt2spk.append(utt + ' ' + spk + '\n')

    with open(f'data/{dataset}_itg3_v5/wav.scp1', 'r') as f:
      for line in f:
        item = line.strip().split()
        dir = item[-1]
        utt = item[0]
        if utt not in utt_delete:
          utt2wav.append(utt + ' ' + dir + '\n')

    with open(f'data/{dataset}_itg3_v5/wav.scp', 'w') as f:
      f.writelines(line for line in utt2wav)

    with open(f'data/{dataset}_itg3_v5/utt2lang', 'w') as f:
      f.writelines(line for line in utt2lang)

    with open(f'data/{dataset}_itg3_v5/utt2spk', 'w') as f:
      f.writelines(line for line in utt2spk)


# unique_spks = []
# with open('data/train_itg/utt2spk') as f:
#   for line in f:
#     spk = line.strip().split()[-1]
#     if spk in unique_spks:
#       continue
#     unique_spks.append(spk)
#
# print(len(unique_spks))


'split -d -l 100 wavs'
def convert(i):
  with open(f'x{i:02d}', 'r') as f:
    path = '/data/pytong/wav/itg_4000/original'
    for line in f:
      wav = line.strip()
      os.system(f"ffmpeg -i {path}/{wav} "
                f"-vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav "
                f"/data/pytong/wav/itg_4000/{wav}")


def hz_stat():
  spk2lang = dict()
  for dataset in ['train', 'lre07']:
    with open(f'/data/pytong/data/{dataset}_itg3/utt2lang', 'r') as f:
      for line in f:
        item = line.strip().split()
        spk = item[0].split('-')[0]
        lang = item[-1]
        spk2lang[spk] = lang

  wavs = os.listdir('.')

  hzdict = dict()
  monodict = dict()
  for wav in wavs:
    item = list(os.popen(f'soxi {wav}'))
    spk = wav.split('-')[0]
    try:
      if spk2lang[spk] == 'uk':
        hz = item[3].strip().split(':')[-1].strip()
        mono = item[2].strip().split(':')[-1].strip()
        if hz not in hzdict.keys():
          hzdict[hz] = [spk]
        else:
          hzdict[hz].append(spk)
        if mono not in monodict.keys():
          monodict[mono] = [spk]
        else:
          monodict[mono].append(spk)
    except:
      pass

'''
len(monodict['1'])
len(monodict['2'])
len(hzdict['11025'])
len(hzdict['12000'])
len(hzdict['22050'])
len(hzdict['24000'])
len(hzdict['48000'])
len(hzdict['96000'])
'''


# first mkdir itg_0429
# total 462
def newdata_0429():
  utt2lang = []
  utt2spk = []
  utt2wav = []
  utt2nid = []
  spk2gender = []
  unique_spks = []
  unique_answers = []
  tag2lang = {'Non Native Heavy': 'nnh', 'Non Native Moderate': 'nnm',
              'Non Native Light': 'nnl', 'Non Native Netural': 'nnn',
              'us': 'us', 'uk': 'uk'}
  workbook = xlrd.open_workbook('data.info/iTG_0429.xlsx')
  for sheetname in ['Heavy', 'Moderate', 'Light', 'Netural', 'Native']:
    sheet = workbook.sheet_by_name(sheetname)
    for i in range(1, sheet.nrows):
      answer_id = str(sheet.row_values(i)[0])
      # name = str(sheet.row_values(i)[1]).lower()
      # for char in ['-', "'", '.']:
      #   spk = name.replace(char, '').replace(' ', '')
      nid = str(sheet.row_values(i)[3]).strip().replace(' ', '')
      gender = 'f' if (sheet.row_values(i)[2] == 1) else 'm'
      original_link = str(sheet.row_values(i)[5]).replace(' ', '%20')
      video_name = original_link.split('/')[-1]
      video_id = video_name.split('.')[0]
      spk = video_id.split('-')[0].split('_')[0]
      utt_id = f'{spk}-{answer_id}'
      tag = str(sheet.row_values(i)[-1])
      if spk not in unique_spks:
        unique_spks.append(spk)
        spk2gender.append(spk + ' ' + gender + '\n')
      if answer_id in unique_answers:
        continue
      if tag == '?':
        continue
      unique_answers.append(answer_id)
      utt2nid.append(utt_id + ' ' + nid + '\n')
      # try:
      #   # os.system(f'wget -O {video_name} {original_link}')
      #   # os.system(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
      #   print(f'wget -O {video_name} {original_link}')
      #   print(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
      # except Exception as e:
      #   print(f'ERROR! {video_name}', e)
      #   continue
      # lang = tag2lang[tag]
      # directory = f'/data/pytong/wav/itg_0429/{video_id}.wav'
      # utt2lang.append(utt_id + ' ' + lang + '\n')
      # utt2spk.append(utt_id + ' ' + spk + '\n')
      # utt2wav.append(utt_id + ' ' + directory + '\n')

  # print('total:', len(utt2spk))
  # random.seed(20)
  # lre07_indices = sorted(random.sample(range(len(utt2wav)), int((len(utt2wav))/10)))
  # print(lre07_indices)
  # train_indices = list(range(len(utt2wav)))
  # for index in lre07_indices:
  #   train_indices.remove(index)
  # print(train_indices)
  # for dataset in ['train', 'lre07']:
  #   with open(f'data/{dataset}_itg0429/wav.scp', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2wav)[locals()[f'{dataset}_indices']]))
  #   with open(f'data/{dataset}_itg0429/utt2lang', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2lang)[locals()[f'{dataset}_indices']]))
  #   with open(f'data/{dataset}_itg0429/utt2spk', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2spk)[locals()[f'{dataset}_indices']]))
  # with open('data/train_itg0429/spk2gender', 'w') as f:
  #   f.writelines(line for line in spk2gender)
  with open('data/utt2nid', 'a') as f:
    f.writelines(line for line in utt2nid)

# answer_id	Accent	full_name	gender	nationality	question_text	answer_video_path
# 0506 0512
def newdata_0506(date='0506'):
  utt2lang = []
  utt2spk = []
  utt2wav = []
  utt2nid = []
  spk2gender = []
  unique_spks = []
  unique_answers = []
  tag2lang = {'Non-native - Heavy': 'nnh', 'Non-native - Moderate': 'nnm',
              'Non-native - Light': 'nnl', 'Non-native - Neutral': 'nnn',
              'Native-American': 'us', 'Native-British': 'uk'}
  workbook = xlrd.open_workbook(f'data.info/iTG_{date}.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    nid = str(sheet.row_values(i)[4]).strip().replace(' ', '')
    gender = 'f' if (sheet.row_values(i)[3] == 1) else 'm'
    original_link = str(sheet.row_values(i)[6]).replace(' ', '%20')
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    tag = str(sheet.row_values(i)[1])
    if spk not in unique_spks:
      unique_spks.append(spk)
      spk2gender.append(spk + ' ' + gender + '\n')
    if answer_id in unique_answers:
      continue
    try:
      lang = tag2lang[tag]
    except KeyError:
      continue
    unique_answers.append(answer_id)
    utt2nid.append(utt_id + ' ' + nid + '\n')

  # with open('data/utt2nid', 'a') as f:
  #   f.writelines(line for line in utt2nid)
    try:
      # os.system(f'wget -O {video_name} {original_link}')
      # os.system(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
      print(f'wget -O {video_name} {original_link}')
      print(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
    except Exception as e:
      print(f'ERROR! {video_name}', e)
      continue

    directory = f'/data/pytong/wav/itg_{date}/{video_id}.wav'
    utt2lang.append(utt_id + ' ' + lang + '\n')
    utt2spk.append(utt_id + ' ' + spk + '\n')
    utt2wav.append(utt_id + ' ' + directory + '\n')
  print('total:', len(utt2spk))
  random.seed(20)
  lre07_indices = sorted(random.sample(range(len(utt2wav)), int((len(utt2wav))/10)))
  print(lre07_indices)
  train_indices = list(range(len(utt2wav)))
  for index in lre07_indices:
    train_indices.remove(index)
  print(train_indices)
  for dataset in ['train', 'lre07']:
    with open(f'data/{dataset}_itg{date}/wav.scp', 'w') as f:
      f.writelines(line for line in list(np.array(utt2wav)[locals()[f'{dataset}_indices']]))
    with open(f'data/{dataset}_itg{date}/utt2lang', 'w') as f:
      f.writelines(line for line in list(np.array(utt2lang)[locals()[f'{dataset}_indices']]))
    with open(f'data/{dataset}_itg{date}/utt2spk', 'w') as f:
      f.writelines(line for line in list(np.array(utt2spk)[locals()[f'{dataset}_indices']]))
  with open(f'data/train_itg{date}/spk2gender', 'w') as f:
    f.writelines(line for line in spk2gender)

# answer_id	Accent 	gender	nationality	question_text	answer_video_path
# 0520 0528 0610 0617 0703 0716 0806 0819
def newdata_0520(date='0520'):
  utt2lang = []
  utt2spk = []
  utt2wav = []
  utt2nid = []
  spk2gender = []
  unique_spks = []
  unique_answers = []
  tag2lang = {'Non-native - Heavy': 'nnh', 'Non-native - Moderate': 'nnm',
              'Non-native - Light': 'nnl', 'Non-native - Neutral': 'nnn',
              'Native-American': 'us', 'Native-British': 'uk', 'None': 'us'}
  workbook = xlrd.open_workbook(f'data.info/iTG_{date}.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    nid = str(sheet.row_values(i)[3]).strip().replace(' ', '')
    gender = 'f' if (sheet.row_values(i)[2] == 1) else 'm'
    original_link = str(sheet.row_values(i)[5]).replace(' ', '%20')
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    tag = str(sheet.row_values(i)[1])
    if spk not in unique_spks:
      unique_spks.append(spk)
      spk2gender.append(spk + ' ' + gender + '\n')
    if answer_id in unique_answers:
      continue
    try:
      lang = tag2lang[tag]
    except KeyError:
      continue
    unique_answers.append(answer_id)
    utt2nid.append(utt_id + ' ' + nid + '\n')

    try:
      # os.system(f'wget -O {video_name} {original_link}')
      # os.system(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
      print(f'wget -O {video_name} {original_link}')
      print(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
    except Exception as e:
      print(f'ERROR! {video_name}', e)
      continue

    directory = f'/data/pytong/wav/itg_{date}/{video_id}.wav'
    utt2lang.append(utt_id + ' ' + lang + '\n')
    utt2spk.append(utt_id + ' ' + spk + '\n')
    utt2wav.append(utt_id + ' ' + directory + '\n')
  print('total:', len(utt2spk))
  with open(f'data/data_itg{date}/wav.scp', 'w') as f:
    f.writelines(line for line in utt2wav)
  with open(f'data/data_itg{date}/utt2lang', 'w') as f:
    f.writelines(line for line in utt2lang)
  with open(f'data/data_itg{date}/utt2spk', 'w') as f:
    f.writelines(line for line in utt2spk)
  with open(f'data/data_itg{date}/spk2gender', 'w') as f:
    f.writelines(line for line in spk2gender)
  # random.seed(20)
  # lre07_indices = sorted(random.sample(range(len(utt2wav)), int((len(utt2wav))/10)))
  # print(lre07_indices)
  # train_indices = list(range(len(utt2wav)))
  # for index in lre07_indices:
  #   train_indices.remove(index)
  # print(train_indices)
  # for dataset in ['train', 'lre07']:
  #   with open(f'data/{dataset}_itg{date}/wav.scp', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2wav)[locals()[f'{dataset}_indices']]))
  #   with open(f'data/{dataset}_itg{date}/utt2lang', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2lang)[locals()[f'{dataset}_indices']]))
  #   with open(f'data/{dataset}_itg{date}/utt2spk', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2spk)[locals()[f'{dataset}_indices']]))
  # with open(f'data/train_itg{date}/spk2gender', 'w') as f:
  #   f.writelines(line for line in spk2gender)
  # with open('data/utt2nid', 'a') as f:
  #   f.writelines(line for line in utt2nid)

# answer_id	Accent	full_name	gender	nationality	question_text	answer_video_path
def newdata_0603():
  utt2lang = []
  utt2spk = []
  utt2wav = []
  utt2nid = []
  spk2gender = []
  unique_spks = []
  unique_answers = []
  tag2lang = {'Non-native - Heavy': 'nnh', 'Non-native - Moderate': 'nnm',
              'Non-native - Light': 'nnl', 'Non-native - Neutral': 'nnn',
              'Native-American': 'us', 'Native-British': 'uk', 'None': 'nnl'}
  workbook = xlrd.open_workbook('data.info/iTG_0603.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    nid = str(sheet.row_values(i)[4]).strip().replace(' ', '')
    gender = 'f' if (sheet.row_values(i)[3] == 1) else 'm'
    original_link = str(sheet.row_values(i)[6]).replace(' ', '%20')
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    tag = str(sheet.row_values(i)[1])
    if spk not in unique_spks:
      unique_spks.append(spk)
      spk2gender.append(spk + ' ' + gender + '\n')
    if answer_id in unique_answers:
      continue
    try:
      lang = tag2lang[tag]
    except KeyError:
      continue
    unique_answers.append(answer_id)
    utt2nid.append(utt_id + ' ' + nid + '\n')

  #   try:
  #     os.system(f'wget -O {video_name} {original_link}')
  #     os.system(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
  #     # print(f'wget -O {video_name} {original_link}')
  #     # print(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
  #   except Exception as e:
  #     print(f'ERROR! {video_name}', e)
  #     continue
  #
  #   directory = f'/data/pytong/wav/itg_0603/{video_id}.wav'
  #   utt2lang.append(utt_id + ' ' + lang + '\n')
  #   utt2spk.append(utt_id + ' ' + spk + '\n')
  #   utt2wav.append(utt_id + ' ' + directory + '\n')
  # print('total:', len(utt2spk))
  # random.seed(20)
  # lre07_indices = sorted(random.sample(range(len(utt2wav)), int((len(utt2wav))/10)))
  # print(lre07_indices)
  # train_indices = list(range(len(utt2wav)))
  # for index in lre07_indices:
  #   train_indices.remove(index)
  # print(train_indices)
  # for dataset in ['train', 'lre07']:
  #   with open(f'data/{dataset}_itg0603/wav.scp', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2wav)[locals()[f'{dataset}_indices']]))
  #   with open(f'data/{dataset}_itg0603/utt2lang', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2lang)[locals()[f'{dataset}_indices']]))
  #   with open(f'data/{dataset}_itg0603/utt2spk', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2spk)[locals()[f'{dataset}_indices']]))
  # with open('data/train_itg0603/spk2gender', 'w') as f:
  #   f.writelines(line for line in spk2gender)
  with open('data/utt2nid', 'a') as f:
    f.writelines(line for line in utt2nid)

# answer_id	Accent	full_name	gender	nationality	question_text
def newdata_0623():
  utt2lang = []
  utt2spk = []
  utt2wav = []
  utt2nid = []
  spk2gender = []
  unique_spks = []
  unique_answers = []
  tag2lang = {'Non-native - Heavy': 'nnh', 'Non-native - Moderate': 'nnm',
              'Non-native - Light': 'nnl', 'Non-native - Neutral': 'nnn',
              'Native-American': 'us', 'Native-British': 'uk'}
  workbook = xlrd.open_workbook('data.info/iTG_0623.xlsx')
  sheet = workbook.sheet_by_name('accent')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    nid = str(sheet.row_values(i)[4]).strip().replace(' ', '')
    gender = 'f' if (sheet.row_values(i)[3] == 1) else 'm'
    original_link = str(sheet.row_values(i)[6]).replace(' ', '%20')
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    tag = str(sheet.row_values(i)[1])
    if spk not in unique_spks:
      unique_spks.append(spk)
      spk2gender.append(spk + ' ' + gender + '\n')
    if answer_id in unique_answers:
      continue
    try:
      lang = tag2lang[tag]
    except KeyError:
      continue
    unique_answers.append(answer_id)
    utt2nid.append(utt_id + ' ' + nid + '\n')
  with open('data/utt2nid', 'a') as f:
    f.writelines(line for line in utt2nid)
  #   try:
  #     os.system(f'wget -O {video_name} {original_link}')
  #     os.system(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
  #     # print(f'wget -O {video_name} {original_link}')
  #     # print(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
  #   except Exception as e:
  #     print(f'ERROR! {video_name}', e)
  #     continue
  #
  #   directory = f'/data/pytong/wav/itg_0623/{video_id}.wav'
  #   utt2lang.append(utt_id + ' ' + lang + '\n')
  #   utt2spk.append(utt_id + ' ' + spk + '\n')
  #   utt2wav.append(utt_id + ' ' + directory + '\n')
  # print('total:', len(utt2spk))
  # random.seed(20)
  # lre07_indices = sorted(random.sample(range(len(utt2wav)), int((len(utt2wav))/10)))
  # print(lre07_indices)
  # train_indices = list(range(len(utt2wav)))
  # for index in lre07_indices:
  #   train_indices.remove(index)
  # print(train_indices)
  # for dataset in ['train', 'lre07']:
  #   with open(f'data/{dataset}_itg0623/wav.scp', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2wav)[locals()[f'{dataset}_indices']]))
  #   with open(f'data/{dataset}_itg0623/utt2lang', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2lang)[locals()[f'{dataset}_indices']]))
  #   with open(f'data/{dataset}_itg0623/utt2spk', 'w') as f:
  #     f.writelines(line for line in list(np.array(utt2spk)[locals()[f'{dataset}_indices']]))
  # with open('data/train_itg0623/spk2gender', 'w') as f:
  #   f.writelines(line for line in spk2gender)


def testdata_180():
  utt2lang = []
  utt2spk = []
  utt2wav = []
  utt2vname = []
  unique_answers = []

  workbook = xlrd.open_workbook('iTG_test180.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[2])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    if answer_id in unique_answers:
      continue
    unique_answers.append(answer_id)
    try:
      os.system(f'wget -O {video_name} {original_link}')
      os.system(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
      # print(f'wget -O {video_name} {original_link}')
      # print(f'ffmpeg -i {video_name} -vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav {video_id}.wav')
    except Exception as e:
      print(f'ERROR! {video_name}', e)
      continue
    directory = f'/data/pytong/wav/itg_test180/{video_id}.wav'
    utt2lang.append(utt_id + ' ' + 'ok' + '\n')
    utt2spk.append(utt_id + ' ' + spk + '\n')
    utt2wav.append(utt_id + ' ' + directory + '\n')
    utt2vname.append(utt_id + ' ' + video_name + '\n')
  print('total:', len(utt2spk))

  with open(f'data/itg_test180/wav.scp', 'w') as f:
    f.writelines(line for line in utt2wav)
  with open(f'data/itg_test180/utt2lang', 'w') as f:
    f.writelines(line for line in utt2lang)
  with open(f'data/itg_test180/utt2spk', 'w') as f:
    f.writelines(line for line in utt2spk)
  with open('data/itg_test180/utt2vname', 'w') as f:
    f.writelines(line for line in utt2vname)


def testdata_180_label():
  utt2lang = []
  unique_answers = []
  tag2lang = {'Non-native - Heavy': 'nnh', 'Non-native - Moderate': 'nnm',
              'Non-native - Light': 'nnl', 'Non-native - Neutral': 'nnn',
              'Native-American': 'us', 'Native-British': 'uk'}
  workbook = xlrd.open_workbook('iTG测试集人工标注.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[-2]).replace(' ', '%20')
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    tag = str(sheet.row_values(i)[1])
    if answer_id in unique_answers:
      continue
    try:
      lang = tag2lang[tag]
    except KeyError:
      continue
    unique_answers.append(answer_id)
    utt2lang.append(utt_id + ' ' + lang + '\n')

  with open('data/itg_test180/utt2lang', 'w') as f:
    f.writelines(line for line in utt2lang)


def relabel_list_til0603():
  to_write = [['utterance', 'gender', 'itg_label', 'relabel', 'link']]
  utt2lang = {}

  with open('data/utt2lang_relabel', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang

  tag2lang0429 = {'Non Native Heavy': 'nnh', 'Non Native Moderate': 'nnm',
                  'us': 'us', 'uk': 'uk'}
  #'Non Native Light': 'nnl', 'Non Native Netural': 'nnn',
  workbook = xlrd.open_workbook('data_csv/iTG_0429.xlsx')
  for sheetname in ['Heavy', 'Moderate', 'Light', 'Netural', 'Native']:
    sheet = workbook.sheet_by_name(sheetname)
    for i in range(1, sheet.nrows):
      answer_id = str(sheet.row_values(i)[0])
      gender = 'f' if (sheet.row_values(i)[2] == 1) else 'm'
      original_link = str(sheet.row_values(i)[5])
      video_name = original_link.split('/')[-1]
      video_id = video_name.split('.')[0]
      spk = video_id.split('-')[0].split('_')[0]
      utt_id = f'{spk}-{answer_id}'
      tag = str(sheet.row_values(i)[-1])

      try:
        lang = tag2lang0429[tag]
      except KeyError:
        continue

      if utt_id not in [x[0] for x in to_write]:
        try:
          relabel = utt2lang[utt_id]
        except KeyError:
          relabel = '-'
        to_write.append([utt_id, gender, lang, relabel, original_link])


  tag2lang = {'Non-native - Heavy': 'nnh', 'Non-native - Moderate': 'nnm',
              'Native-American': 'us', 'Native-British': 'uk'}
  #'Non-native - Light': 'nnl', 'Non-native - Neutral': 'nnn'
  workbook = xlrd.open_workbook('data_csv/iTG_0506.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    gender = 'f' if (sheet.row_values(i)[2] == 1) else 'm'
    original_link = str(sheet.row_values(i)[6])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    tag = str(sheet.row_values(i)[1])

    try:
      lang = tag2lang[tag]
    except KeyError:
      continue

    if utt_id not in [x[0] for x in to_write]:
      try:
        relabel = utt2lang[utt_id]
      except KeyError:
        relabel = '-'
      to_write.append([utt_id, gender, lang, relabel, original_link])

  workbook = xlrd.open_workbook('data_csv/iTG_0512.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    gender = 'f' if (sheet.row_values(i)[4] == 1) else 'm'
    original_link = str(sheet.row_values(i)[6])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    tag = str(sheet.row_values(i)[1])

    try:
      lang = tag2lang[tag]
    except KeyError:
      continue

    if utt_id not in [x[0] for x in to_write]:
      try:
        relabel = utt2lang[utt_id]
      except KeyError:
        relabel = '-'
      to_write.append([utt_id, gender, lang, relabel, original_link])


  workbook = xlrd.open_workbook('data_csv/iTG_0520.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    gender = 'f' if (sheet.row_values(i)[2] == 1) else 'm'
    original_link = str(sheet.row_values(i)[5])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    tag = str(sheet.row_values(i)[1])

    try:
      lang = tag2lang[tag]
    except KeyError:
      continue

    if utt_id not in [x[0] for x in to_write]:
      try:
        relabel = utt2lang[utt_id]
      except KeyError:
        relabel = '-'
      to_write.append([utt_id, gender, lang, relabel, original_link])


  workbook = xlrd.open_workbook('data_csv/iTG_0528.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    gender = 'f' if (sheet.row_values(i)[2] == 1) else 'm'
    original_link = str(sheet.row_values(i)[5])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    tag = str(sheet.row_values(i)[1])

    try:
      lang = tag2lang[tag]
    except KeyError:
      continue

    if utt_id not in [x[0] for x in to_write]:
      try:
        relabel = utt2lang[utt_id]
      except KeyError:
        relabel = '-'
      to_write.append([utt_id, gender, lang, relabel, original_link])

  workbook = xlrd.open_workbook('data_csv/iTG_0603.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    gender = 'f' if (sheet.row_values(i)[3] == 1) else 'm'
    original_link = str(sheet.row_values(i)[6])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    tag = str(sheet.row_values(i)[1])

    try:
      lang = tag2lang[tag]
    except KeyError:
      continue

    if utt_id not in [x[0] for x in to_write]:
      try:
        relabel = utt2lang[utt_id]
      except KeyError:
        relabel = '-'
      to_write.append([utt_id, gender, lang, relabel, original_link])

  print(len(to_write))
  with open('relabel_list_til0603.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(line for line in to_write)

def utt2lang_edited():
  utt2lang = []
  workbook = xlrd.open_workbook('correct_0530_itg_2cls.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(0, sheet.nrows):
    utt_id = str(sheet.row_values(i)[0])
    tag = str(sheet.row_values(i)[1])
    utt2lang.append(utt_id + ' ' + tag + '\n')

  workbook = xlrd.open_workbook('error_0530_itg_2cls.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    utt_id = str(sheet.row_values(i)[0])
    if not sheet.row_values(i)[3]:
      tag = str(sheet.row_values(i)[1])
    else:
      tag = str(sheet.row_values(i)[3])
    utt2lang.append(utt_id + ' ' + tag + '\n')
  with open('data/lre07_0530_itg/utt2lang_edited', 'w') as f:
    f.writelines(line for line in utt2lang)

def delete_duplicates():
  utt_list = []
  utt2wav = []
  with open('wav.scp.til0716', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      directory = item[-1]
      if utt in utt_list:
        print(utt)
        continue
      utt_list.append(utt)
      utt2wav.append(utt + ' ' + directory + '\n')
  print(len(utt2wav))
  with open('wav.scp.til0716', 'w') as f:
    f.writelines(line for line in utt2wav)

def filter_test():
  utt_list = []
  utt2wav = []
  utt2spk = []
  with open("wav.scp", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      if utt in utt_list:
        print(utt)
      utt_list.append(utt)
  print(len(utt_list))
  with open('utt2lang6.v9', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      spk = utt.split('-')[0]
      directory = item[-1]
      if utt in utt_list:
        utt2wav.append(utt + ' ' + directory + '\n')
        utt2spk.append(utt + ' ' + spk + '\n')
  # with open('spk2gender.bak', 'r') as f:
  #   for line in f:
  #     item = line.strip().split()
  #     utt = item[0]
  #     spk = utt.split('-')[0]
  #     directory = item[-1]
  #     if utt in utt_list:
  #       utt2wav.append(utt + ' ' + directory + '\n')
  #       utt2spk.append(utt + ' ' + spk + '\n')
  print(len(utt2wav))
  print(len(utt2spk))
  with open('utt2lang', 'w') as f:
    f.writelines(line for line in utt2wav)
  with open('utt2spk', 'w') as f:
    f.writelines(line for line in utt2spk)

def check_not_found():
  utt_list = []
  not_found = []
  utt2wav = []
  utt2spk = []
  with open("wav.scp.bak", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      utt_list.append(utt)
  with open('utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      spk = utt.split('-')[0]
      directory = item[-1]
      if utt not in utt_list:
        not_found.append(utt)
  print(not_found)

def two_classes():
  utt2lang = []
  with open('utt2lang.6', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      if lang in ['nnm', 'nnh', 'nn', 'nnl']:
        lang = 'nn'
      else:
        lang = 'ok'
      utt2lang.append(utt + ' ' + lang + '\n')
  print(len(utt2lang))
  with open(f'utt2lang.2.nnl', 'w') as f:
    f.writelines(line for line in utt2lang)

def four_classes(f='utt2lang.6'):
  map4 = {
    'us': 'us',
    'uk': 'uk',
    'nnh': 'nnh',
    'nnm': 'nnh',
    'nnn': 'nnl',
    'nnl': 'nnl'
  }
  utt2lang = []
  with open(f, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      try:
        lang = map4[lang]
      except KeyError:
        continue
      utt2lang.append(utt + ' ' + lang + '\n')
  print(len(utt2lang))
  with open(f'utt2lang.4.nnl+nnn', 'w') as f:
    f.writelines(line for line in utt2lang)


def five_classes():
  utt2lang = []
  with open('utt2lang.6', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      if lang in ['nnm', 'nnh', 'nn']:
        lang = 'nn'
      utt2lang.append(utt + ' ' + lang + '\n')
  with open(f'utt2lang.5', 'w') as f:
    f.writelines(line for line in utt2lang)

def two_classes_no_nnl():
  utt2lang = []
  with open('utt2lang.6', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      if lang == 'nnl':
        continue
      if lang in ['nnm', 'nnh', 'nn']:
        lang = 'nn'
      elif lang != 'nnl':
        lang = 'ok'
      utt2lang.append(utt + ' ' + lang + '\n')
  print(len(utt2lang))
  with open(f'utt2lang.2.nonnl', 'w') as f:
    f.writelines(line for line in utt2lang)

def two_classes_3and3():
  utt2lang = []
  # with open('utt2lang.6', 'r') as f:
  with open('utt2lang_relabel_itg_446', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      if lang in ['nnm', 'nnh', 'nn', 'nnl']:
        lang = 'nn'
      else:
        lang = 'ok'
      utt2lang.append(utt + ' ' + lang + '\n')
  with open(f'utt2lang.2.3and3', 'w') as f:
    f.writelines(line for line in utt2lang)


def select_testset_446():
  utt2lang = {}
  with open('utt2lang3', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      utt2lang[utt] = lang

  to_write = []
  us = 0
  uk = 0
  nnl = 0
  nnn = 0
  nnm = 0
  with open('data/utt2lang_relabel_0607', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      if len(to_write) > 450:
        break
      if utt in utt2lang.keys():
        continue
      if lang == 'nn':
        to_write.append(utt + ' ' + lang + '\n')
      if lang == 'us' and us < 180:
        us += 1
        to_write.append(utt + ' ' + lang + '\n')
      if lang == 'uk' and uk < 60:
        uk += 1
        to_write.append(utt + ' ' + lang + '\n')
      if lang == 'nnl' and nnl < 60:
        nnl += 1
        to_write.append(utt + ' ' + lang + '\n')
      if lang == 'nnm' and nnm < 60:
        nnm += 1
        to_write.append(utt + ' ' + lang + '\n')
      if lang == 'nnn' and nnn < 60:
        nnn += 1
        to_write.append(utt + ' ' + lang + '\n')
  print(len(to_write))
  with open(f'utt2lang_test450', 'w') as f:
    f.writelines(line for line in to_write)

def select_testset_mturk_180():
  to_write = []
  train = []
  us = 0
  uk = 0
  nnl = 0
  nnn = 0
  nnm = 0
  nnh = 0
  with open('mturk/utt2lang6.v3', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      if lang == 'us' and us < 30:
        us += 1
        to_write.append(utt + ' ' + lang + '\n')
      elif lang == 'uk' and uk < 30:
        uk += 1
        to_write.append(utt + ' ' + lang + '\n')
      elif lang == 'nnl' and nnl < 30:
        nnl += 1
        to_write.append(utt + ' ' + lang + '\n')
      elif lang == 'nnm' and nnm < 30:
        nnm += 1
        to_write.append(utt + ' ' + lang + '\n')
      elif lang == 'nnn' and nnn < 30:
        nnn += 1
        to_write.append(utt + ' ' + lang + '\n')
      elif lang == 'nnh' and nnh < 30:
        nnh += 1
        to_write.append(utt + ' ' + lang + '\n')
      else:
        train.append(utt + ' ' + lang + '\n')
  print(len(to_write), len(train))
  with open(f'utt2lang_mturk_test180', 'w') as f:
    f.writelines(line for line in to_write)
  with open(f'utt2lang_mturk_train3535', 'w') as f:
    f.writelines(line for line in train)

def filter_vivian():
  utt_list = []
  utt2lang = []
  with open("wav.scp", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      utt_list.append(utt)
  with open('utt2lang_vivian', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if utt in utt_list:
        utt2lang.append(utt + ' ' + lang + '\n')
  print(len(utt2lang))
  with open('utt2lang_vivian.train', 'w') as f:
    f.writelines(line for line in utt2lang)

import contextlib
import wave
# import librosa
def duration():
  res = {}
  res_list = []
  error_list = {}
  with open("wav.scp", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      path = item[-1]
      try:
        with contextlib.closing(wave.open(path, 'r')) as wav:
          frames = wav.getnframes()
          rate = wav.getframerate()
          duration = frames / float(rate)
          res[utt] = duration
          res_list.append(float(duration))
      except Exception as e:
        error_list[utt] = e
        continue
  return res, res_list, error_list
# len([r for r in res_list if r<10]) 17


def _check_wav_power():
  res = {}
  res_list = []
  error_list = {}
  with open("wav.scp", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      path = item[-1]
      try:
        wav_data, sr = librosa.load(path, sr=None)
      except Exception as e:
        error_list[utt] = e
        continue
      energy = librosa.feature.rms(y=wav_data, hop_length=int(0.010 * sr))[0]
      avg = sum(energy)/len(energy)
      res[utt] = avg
      res_list.append(float(avg))
  return res, res_list, error_list
# len([r for r in res_list if r<=0.003]) 63

import parselmouth
def noise(fn):
  try:
    snd = parselmouth.Sound(fn)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    proportion = len(pitch_values[pitch_values > 0]) / len(pitch_values)
  except Exception as e:
    print(e)
    return 0
  return proportion

def _check_noise():
  res = {}
  res_list = []
  error_list = {}
  count = 0
  with open("wav.scp", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      path = item[-1]
      count += 1
      print(count)
      print(utt)
      try:
        snd = parselmouth.Sound(path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        proportion = len(pitch_values[pitch_values > 0]) / len(pitch_values)
      except Exception as e:
        error_list[utt] = e
        continue
      res[utt] = proportion
      res_list.append(float(proportion))
  return res, res_list, error_list

def utt2lang_1800_0610_0617():
  utt2lang = []
  wrong_utt = []
  with open("/data/pytong/kaldi/egs/lre07/v1/error_0617_exp29", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      wrong_utt.append(utt)
  with open("/data/pytong/kaldi/egs/lre07/v1/error_0610_exp29", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      wrong_utt.append(utt)
  with open("utt2lang.6.bak", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if utt in wrong_utt:
        continue
      utt2lang.append(utt + ' ' + lang + '\n')
  with open('utt2lang.6.filtered', 'w') as f:
    f.writelines(line for line in utt2lang)

def utt2lang_1800_0617():
  utt2lang = []
  wrong_utt = []
  utt_0610 = []
  with open("/data/pytong/kaldi/egs/lre07/v1/error_0617_exp29", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      wrong_utt.append(utt)
  with open("/data/pytong/data/data_0610_test29/utt2lang_0610data_relabel_itg.2", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      utt_0610.append(utt)
  with open("bak/utt2lang.6.bak", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if utt in wrong_utt:
        continue
      utt2lang.append(utt + ' ' + lang + '\n')
  with open('utt2lang.6.1800_0617', 'w') as f:
    f.writelines(line for line in utt2lang)


def data_balance():
  us = 279
  uk = 147
  nnn = 152
  nnl = 152
  utt2lang = []
  with open('bak/utt2lang.6.filtered', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if lang in ['ok', 'nn', 'nnm', 'nnh']:
        utt2lang.append(utt + ' ' + lang + '\n')
      if lang == 'us':
        if us > 0:
          us -= 1
          utt2lang.append(utt + ' ' + lang + '\n')
      if lang == 'uk':
        if uk > 0:
          uk -= 1
          utt2lang.append(utt + ' ' + lang + '\n')
      if lang == 'nnl':
        if nnl > 0:
          nnl -= 1
          utt2lang.append(utt + ' ' + lang + '\n')
      if lang == 'nnn':
        if nnn > 0:
          nnn -= 1
          utt2lang.append(utt + ' ' + lang + '\n')
  print(len(utt2lang))
  with open('utt2lang.6.balanced', 'w') as f:
    f.writelines(line for line in utt2lang)

def read_utt2lang(utt2_f):
  labels = {}
  with open(utt2_f) as f:
    line = f.readline()
    while line:
      words = line.replace('\n', '').split(' ')
      labels[words[0]] = words[1]
      line = f.readline()
  return labels

def random_select(utt2_f):
  labels = read_utt2lang(utt2_f)
  train = open('train_utt2lang', 'w')
  test = open('test_utt2lang', 'w')
  count1 = 0
  count2 = 0
  for k, v in labels.items():
    if random.random() < 0.1:
      count1+=1
      test.write(k + ' ' + v + '\n')
    else:
      count2 +=1
      train.write(k + ' ' + v + '\n')
  train.close()
  test.close()
  print(str(count1)+": "+str(count2))

# random_select('utt2lang6.v2')

def new_utt2nid():
  utt2nid = {}
  to_write = []
  with open("data/utt2nid", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nid = item[-1]
      utt2nid[utt] = nid
  with open("mturk/utt2lang_mturk_train5641", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nid = utt2nid[utt]
      to_write.append(utt + ' ' + nid + '\n')
  with open('mturk/utt2nid_mturk_train5641', 'w') as f:
    f.writelines(line for line in to_write)

def scp_audio():
  utt2wav = []
  with open('data/test_mturk_300/wav.scp', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      path = item[1]
      os.system(f'scp -P 39002 buyingtong@ml.pingan-labs.us:{path} test_mturk_300/.')
      video_name = path.split('/')[-1]
      new_path = f'/Users/buyingtong/User/Paii/asr/paii_lid/test_mturk_300/{video_name}'
      utt2wav.append(utt + ' ' + new_path + '\n')
  with open(f'/Users/buyingtong/User/Paii/nlp/repos/nlp_team/speech/accent_dection/data/wav.scp.test300', 'w') as f:
    f.writelines(line for line in utt2wav)

if __name__ == '__main__':
  # i = int(sys.argv[1])
  # convert(i)
  # nations = get_nation()
  # testdata_180()
  # newdata_0623()
  # testdata_180_label()
  # newdata_0716()
  # select_testset_mturk_180()
  newdata_0520('0826')
  # new_utt2nid()