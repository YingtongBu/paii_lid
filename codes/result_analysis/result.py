import os
import numpy as np
import xlrd
import random

def error():
  map4 = {
    'us': 'us',
    'uk': 'uk',
    'nnh': 'nnh',
    'nnm': 'nnh',
    'nnn': 'nnl',
    'nnl': 'nnh'
  }
  gold = {}
  error = []
  with open('/data/pytong/data/test_mturk_v2/utt2lang.4', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      gold[utt] = lang
  with open('exp_34/ivectors_test_mturk_v2/output', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      # lang2 = gold[utt]
      try:
        lang2 = map4[gold[utt]]
      except KeyError:
        continue
      # if utt not in gold.keys():
      #   error.append([utt, 'no_result', lang])
      # elif lang != gold[utt]:
      #   error.append([utt, gold[utt], lang])
      # if gold[utt] in ['nnm', 'nnh', 'nnl']:
      #   lang2 = 'nn'
      # else:
      #   lang2 = 'ok'
      if lang != lang2: #and lang == 'nn' and 'ytb' not in utt:
        error.append([utt, gold[utt], lang])
  print(len(error))
  with open('error_testdata_exp34.4', 'w') as f:
    # f.writelines(line[0] + ' ' + line[1] + ' ' + '\n' for line in error)
    f.writelines(line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + '\n' for line in error)

def correct():
  gold = {}
  correct = []
  with open('data/data_0610_test/utt2lang.6', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      gold[utt] = lang
  with open('exp_28/ivectors_data_0610_test/output', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if gold[utt] in ['nnm', 'nnh', 'nn']:
        lang2 = 'nn'
      else:
        lang2 = 'ok'
      if lang == lang2 and 'ytb' not in utt:
        correct.append([utt, gold[utt]])
  with open('utt2lang', 'w') as f:
    f.writelines(line[0] + ' ' + line[1] + ' ' + '\n' for line in correct)


def error5to2():
  gold = {}
  error = []
  with open('data/lre07_0530_relabel0604/utt2lang.5', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      gold[utt] = lang
  with open('exp_25/ivectors_lre07_0530_relabel0604/output', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      lang1, lang2 = 'nn', 'nn'
      if lang != 'nn':
        lang1 = 'ok'
      if gold[utt] != 'nn':
        lang2 = 'ok'
      if lang1 != lang2:
        error.append([utt, gold[utt], lang])
  with open('error_0530_relabel0604_5to2cls_', 'w') as f:
    f.writelines(line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + '\n' for line in error)

def error6to2():
  gold = {}
  error = []
  with open('data/lre07_0530_itg/utt2lang.6', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      gold[utt] = lang
  with open('exp_23/ivectors_lre07_0530_itg_200/output', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if lang in ['nnm', 'nnh']:
        lang1 = 'nn'
      else:
        lang1 = 'ok'
      if gold[utt] in ['nnm', 'nnh']:
        lang2 = 'nn'
      else:
        lang2 = 'ok'
      if lang1 != lang2:
        error.append([utt, gold[utt], lang])
  with open('error_0530_itg_6to2cls_i200', 'w') as f:
    f.writelines(line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + '\n' for line in error)


def errorbyposteriors(threshold):
  gold = {}
  pred = {}
  error = []
  with open('data/train_0530_youtube/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      gold[utt] = lang
  with open('exp_24/ivectors_train_0530_youtube/output', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      pred[utt] = lang
      # if lang != gold[utt] and 'ytb' not in utt:
      #   error.append([utt, gold[utt], lang])
  with open('exp_24/ivectors_train_0530_youtube/posteriors', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      if pred[utt] != gold[utt]:
        posteriors = max(float(item[2]), float(item[3]), float(item[4]))
        if posteriors > threshold:
          error.append([utt, gold[utt], pred[utt]])
  print(len(error))
  with open('error_0530_youtube_posteriors_filtered', 'w') as f:
    f.writelines(line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + '\n' for line in error)

def error_compare():
  youtube_lang = {}
  youtube_list = []
  same_pred_list = []
  final_list = []
  final2_list = []
  final3_list = []
  with open('error_0530_relabel0604_5to2cls_', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      lang = item[-1]
      youtube_list.append(utt)
      youtube_lang[utt] = lang
  with open('error_0530_relabel0604_2cls', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      lang = item[-1]
      if utt in youtube_list:
        final_list.append(utt)
  print(len(final_list))
      # try:
      #   if lang == youtube_lang[utt]:
      #     same_pred_list.append(utt)
      # except KeyError:
      #   pass
  # with open('results/0530/error_0530_itg_2cls', 'r') as f:
  #   for line in f:
  #     item = line.split()
  #     utt = item[0]
  #     if utt in final_list:
  #       final2_list.append(utt)
  # with open('results/0530/error_0530_itg_6to2cls', 'r') as f:
  #   for line in f:
  #     item = line.split()
  #     utt = item[0]
  #     if utt in final2_list:
  #       final3_list.append(utt)
  print(len(final3_list))
  print(len(same_pred_list))


import csv
def error_link():
  to_write = []
  utt2gold = {}
  utt2pred = {}
  with open('error_testdata_exp31', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      gold = item[1]
      pred = item[2]
      utt2gold[utt] = gold
      utt2pred[utt] = pred
  print(len(utt2gold))

  # with open('relabel_0608', 'r') as f:
  #   for line in f:
  #     item = line.split()
  #     utt = item[0]
  #     try:
  #       lang = item[3]
  #       try:
  #         utt2gold[utt] = lang
  #       except KeyError:
  #         continue
  #     except IndexError:
  #       continue
  #
  #     # utt2pred[utt] = pred

  # workbook = xlrd.open_workbook('data.info/iTG_0429.xlsx')
  # for sheetname in ['Heavy', 'Moderate', 'Light', 'Netural', 'Native']:
  #   sheet = workbook.sheet_by_name(sheetname)
  #   for i in range(1, sheet.nrows):
  #     answer_id = str(sheet.row_values(i)[0])
  #     original_link = str(sheet.row_values(i)[5])
  #     video_name = original_link.split('/')[-1]
  #     video_id = video_name.split('.')[0]
  #     spk = video_id.split('-')[0].split('_')[0]
  #     utt_id = f'{spk}-{answer_id}'
  #     try:
  #       if utt_id not in [x[0] for x in to_write]:
  #         # to_write.append([utt_id, utt2gold[utt_id], original_link])
  #         to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
  #     except KeyError:
  #       continue
  #
  # workbook = xlrd.open_workbook('data.info/iTG_0506.xlsx')
  # sheet = workbook.sheet_by_name('Sheet1')
  # for i in range(1, sheet.nrows):
  #   answer_id = str(sheet.row_values(i)[0])
  #   original_link = str(sheet.row_values(i)[6])
  #   video_name = original_link.split('/')[-1]
  #   video_id = video_name.split('.')[0]
  #   spk = video_id.split('-')[0].split('_')[0]
  #   utt_id = f'{spk}-{answer_id}'
  #   try:
  #     if utt_id not in [x[0] for x in to_write]:
  #       # to_write.append([utt_id, utt2gold[utt_id], original_link])
  #       to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
  #   except KeyError:
  #     continue
  #
  # workbook = xlrd.open_workbook('data.info/iTG_0512.xlsx')
  # sheet = workbook.sheet_by_name('Sheet1')
  # for i in range(1, sheet.nrows):
  #   answer_id = str(sheet.row_values(i)[0])
  #   original_link = str(sheet.row_values(i)[6])
  #   video_name = original_link.split('/')[-1]
  #   video_id = video_name.split('.')[0]
  #   spk = video_id.split('-')[0].split('_')[0]
  #   utt_id = f'{spk}-{answer_id}'
  #   try:
  #     if utt_id not in [x[0] for x in to_write]:
  #       # to_write.append([utt_id, utt2gold[utt_id], original_link])
  #       to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
  #   except KeyError:
  #     continue
  #
  # workbook = xlrd.open_workbook('data.info/iTG_0520.xlsx')
  # sheet = workbook.sheet_by_name('Sheet1')
  # for i in range(1, sheet.nrows):
  #   answer_id = str(sheet.row_values(i)[0])
  #   original_link = str(sheet.row_values(i)[5])
  #   video_name = original_link.split('/')[-1]
  #   video_id = video_name.split('.')[0]
  #   spk = video_id.split('-')[0].split('_')[0]
  #   utt_id = f'{spk}-{answer_id}'
  #   try:
  #     if utt_id not in [x[0] for x in to_write]:
  #       # to_write.append([utt_id, utt2gold[utt_id], original_link])
  #       to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
  #   except KeyError:
  #     continue
  #
  # workbook = xlrd.open_workbook('data.info/iTG_0528.xlsx')
  # sheet = workbook.sheet_by_name('Sheet1')
  # for i in range(1, sheet.nrows):
  #   answer_id = str(sheet.row_values(i)[0])
  #   original_link = str(sheet.row_values(i)[5])
  #   video_name = original_link.split('/')[-1]
  #   video_id = video_name.split('.')[0]
  #   spk = video_id.split('-')[0].split('_')[0]
  #   utt_id = f'{spk}-{answer_id}'
  #   try:
  #     if utt_id not in [x[0] for x in to_write]:
  #       # to_write.append([utt_id, utt2gold[utt_id], original_link])
  #       to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
  #   except KeyError:
  #     continue
  #
  # workbook = xlrd.open_workbook('data.info/iTG_0603.xlsx')
  # sheet = workbook.sheet_by_name('Sheet1')
  # for i in range(1, sheet.nrows):
  #   answer_id = str(sheet.row_values(i)[0])
  #   original_link = str(sheet.row_values(i)[6])
  #   video_name = original_link.split('/')[-1]
  #   video_id = video_name.split('.')[0]
  #   spk = video_id.split('-')[0].split('_')[0]
  #   utt_id = f'{spk}-{answer_id}'
  #
  #   try:
  #     if utt_id not in [x[0] for x in to_write]:
  #       # to_write.append([utt_id, utt2gold[utt_id], original_link])
  #       to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
  #   except KeyError:
  #     continue

  # workbook = xlrd.open_workbook('data.info/iTG_0610.xlsx')
  # sheet = workbook.sheet_by_name('Sheet1')
  # for i in range(1, sheet.nrows):
  #   answer_id = str(sheet.row_values(i)[0])
  #   original_link = str(sheet.row_values(i)[5])
  #   video_name = original_link.split('/')[-1]
  #   video_id = video_name.split('.')[0]
  #   spk = video_id.split('-')[0].split('_')[0]
  #   utt_id = f'{spk}-{answer_id}'
  #
    # try:
    #   if utt_id not in [x[0] for x in to_write]:
    #     # to_write.append([utt_id, utt2gold[utt_id], original_link])
    #     to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
    # except KeyError:
    #   continue

  # workbook = xlrd.open_workbook('data.info/iTG_0617.xlsx')
  # sheet = workbook.sheet_by_name('Sheet1')
  # for i in range(1, sheet.nrows):
  #   answer_id = str(sheet.row_values(i)[0])
  #   original_link = str(sheet.row_values(i)[5])
  #   video_name = original_link.split('/')[-1]
  #   video_id = video_name.split('.')[0]
  #   spk = video_id.split('-')[0].split('_')[0]
  #   utt_id = f'{spk}-{answer_id}'
  #   try:
  #     if utt_id not in [x[0] for x in to_write]:
  #       to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
  #   except KeyError:
  #     continue

  workbook = xlrd.open_workbook('data.info/iTG_test180.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[2])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    try:
      if utt_id not in [x[0] for x in to_write]:
        to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
    except KeyError:
      continue

  print(len(to_write))
  with open('error_testdata_exp31.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(line for line in to_write)

def error_link2():
  to_write = []
  utt2gold = {}
  utt2pred = {}
  youtube_lang = {}
  youtube_list = []
  with open('results/0530/error_0530_youtube_6cls', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      lang = item[-1]
      youtube_list.append(utt)
      youtube_lang[utt] = lang
  with open('results/0530/error_0530_itg_6cls', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      gold = item[1]
      lang = item[-1]
      if utt not in youtube_list:
          utt2gold[utt] = gold
          utt2pred[utt] = lang

  workbook = xlrd.open_workbook('data_csv/iTG_0506.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[6])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    try:
      if utt_id not in [x[0] for x in to_write]:
        to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
    except KeyError:
      continue

  workbook = xlrd.open_workbook('data_csv/iTG_0520.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[5])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    try:
      if utt_id not in [x[0] for x in to_write]:
        to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
    except KeyError:
      continue

  workbook = xlrd.open_workbook('data_csv/iTG_0512.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[6])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    try:
      if utt_id not in [x[0] for x in to_write]:
        to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
    except KeyError:
      continue

  workbook = xlrd.open_workbook('data_csv/iTG_0429.xlsx')
  for sheetname in ['Heavy', 'Moderate', 'Light', 'Netural', 'Native']:
    sheet = workbook.sheet_by_name(sheetname)
    for i in range(1, sheet.nrows):
      answer_id = str(sheet.row_values(i)[0])
      original_link = str(sheet.row_values(i)[5])
      video_name = original_link.split('/')[-1]
      video_id = video_name.split('.')[0]
      spk = video_id.split('-')[0].split('_')[0]
      utt_id = f'{spk}-{answer_id}'
      try:
        if utt_id not in [x[0] for x in to_write]:
          to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
      except KeyError:
        continue

  workbook = xlrd.open_workbook('data_csv/iTG_0528.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[5])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    try:
      if utt_id not in [x[0] for x in to_write]:
        to_write.append([utt_id, utt2gold[utt_id], utt2pred[utt_id], original_link])
    except KeyError:
      continue

  print(len(to_write))
  with open('error_0530_6cls_itg_only.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(line for line in to_write)

def utt2link():
  utt2link = []
  unique_utt = []

  workbook = xlrd.open_workbook('data.info/iTG_0429.xlsx')
  for sheetname in ['Heavy', 'Moderate', 'Light', 'Netural', 'Native']:
    sheet = workbook.sheet_by_name(sheetname)
    for i in range(1, sheet.nrows):
      answer_id = str(sheet.row_values(i)[0])
      original_link = str(sheet.row_values(i)[5])
      video_name = original_link.split('/')[-1]
      video_id = video_name.split('.')[0]
      spk = video_id.split('-')[0].split('_')[0]
      utt_id = f'{spk}-{answer_id}'
      if utt_id in unique_utt:
        continue
      unique_utt.append(utt_id)
      utt2link.append(utt_id + ' ' + original_link + '\n')

  workbook = xlrd.open_workbook('data.info/iTG_0506.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[6])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    if utt_id in unique_utt:
      continue
    unique_utt.append(utt_id)
    utt2link.append(utt_id + ' ' + original_link + '\n')

  workbook = xlrd.open_workbook('data.info/iTG_0512.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[6])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    if utt_id in unique_utt:
      continue
    unique_utt.append(utt_id)
    utt2link.append(utt_id + ' ' + original_link + '\n')

  workbook = xlrd.open_workbook('data.info/iTG_0520.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[5])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    if utt_id in unique_utt:
      continue
    unique_utt.append(utt_id)
    utt2link.append(utt_id + ' ' + original_link + '\n')

  workbook = xlrd.open_workbook('data.info/iTG_0528.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[5])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    if utt_id in unique_utt:
      continue
    unique_utt.append(utt_id)
    utt2link.append(utt_id + ' ' + original_link + '\n')

  workbook = xlrd.open_workbook('data.info/iTG_0603.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[6])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    if utt_id in unique_utt:
      continue
    unique_utt.append(utt_id)
    utt2link.append(utt_id + ' ' + original_link + '\n')

  workbook = xlrd.open_workbook('data.info/iTG_0610.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[5])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    if utt_id in unique_utt:
      continue
    unique_utt.append(utt_id)
    utt2link.append(utt_id + ' ' + original_link + '\n')

  print(len(utt2link))
  with open('utt2link_til0610', 'w') as f:
    f.writelines(line for line in utt2link)


import numpy as np
def read_ivectors(filepath):
  utt2lang = {}
  with open('data/train_0530_itg/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
  with open('data/lre07_0530_itg/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
  x_us = []
  x_uk = []
  x_nnl = []
  x_nnn = []
  x_nnm = []
  x_nnh = []
  files = os.listdir(filepath)
  for file in files:
    with open(f'{filepath}/{file}', 'r') as f:
      for line in f:
        item = line.strip().split()
        utt = item[0]
        label = utt2lang[utt]
        vector = [float(x) for x in item[1:][1:-1]]
        if label == 'us':
          x_us.append(vector)
        elif label == 'uk':
          x_uk.append(vector)
        elif label == 'nnl':
          x_nnl.append(vector)
        elif label == 'nnn':
          x_nnn.append(vector)
        elif label == 'nnm':
          x_nnm.append(vector)
        elif label == 'nnh':
          x_nnh.append(vector)
  x_nn = x_nnl + x_nnn + x_nnm + x_nnh
  classes = ['us', 'uk', 'nnl', 'nnn', 'nnm', 'nnh', 'nn']
  print(x_uk[490])
  print(np.sum(np.square(np.array(x_us[0]) - np.array(x_nnh[46]))))
  for cls in classes:
    locals()[f'x_{cls}_mean'] = np.mean(np.array(locals()[f'x_{cls}']), axis=0)
    # print(locals()[f'x_{cls}_mean'])
    locals()[f'x_{cls}_var'] = np.mean(np.var(np.array(locals()[f'x_{cls}']), axis=0))
    # print(locals()[f'x_{cls}_var'])
  dist = [[0 for _ in range(len(classes))] for _ in range(len(classes))]
  for i in range(len(classes)):
    a = locals()[f'x_{classes[i]}_mean']
    for j in range(len(classes)):
      b = locals()[f'x_{classes[j]}_mean']
      dist[i][j] = np.sum(np.square(np.array(a) - np.array(b)))
  return dist


def relabel():
  utt2lang = {}
  to_write = []
  with open('data_til0603_relabel0605/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
  with open('data_0530_relabel0604/utt2lang_relable', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if lang != utt2lang[utt]:
        utt2lang[utt] = lang
  with open('relabel0605', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      try:
        lang = item[3]
      except IndexError:
        continue
      if lang == 'delete':
        del utt2lang[utt]
      elif lang != utt2lang[utt]:
        utt2lang[utt] = lang
  for utt in utt2lang.keys():
    to_write.append(utt + ' ' + utt2lang[utt] + '\n')
  with open('data_til0603_relabel0605/utt2lang_relabel', 'w') as f:
    f.writelines(line for line in to_write)


def clean_label():
  original_utt2lang = {}
  relabel_utt2lang = []
  with open('data/train_0530_itg/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      original_utt2lang[utt] = lang
  with open('data/lre07_0530_itg/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      original_utt2lang[utt] = lang
  with open('data/train_itg0603/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      original_utt2lang[utt] = lang
  with open('data/lre07_itg0603/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      original_utt2lang[utt] = lang
  with open('data/utt2lang_relabel', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      try:
        original = original_utt2lang[utt]
      except KeyError:
        continue

      if lang == original:
        relabel_lang = original
      elif lang == 'nn':
        if original in ['nnm', 'nnh']:
          relabel_lang = original
        else:
          relabel_lang = 'nn'
      elif lang == 'ok':
        if original in ['nnn', 'nnl', 'us', 'uk']:
          relabel_lang = original
        else:
          relabel_lang = 'ok'
      relabel_utt2lang.append(utt + ' ' + relabel_lang + '\n')
  print(len(relabel_utt2lang))
  with open('data/utt2lang_relabel_cleaned', 'w') as f:
    f.writelines(line for line in relabel_utt2lang)

def combine_relabel_googlesheets():
  utt2lang = {}
  to_write = []
  with open('data/utt2lang_relabel_cleaned', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
  with open('data/google_sheets_0607', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      try:
        lang = item[2]
      except IndexError:
        continue
      if lang != utt2lang[utt]:
        utt2lang[utt] = lang
  for utt in utt2lang.keys():
    to_write.append(utt + ' ' + utt2lang[utt] + '\n')
  with open('data/utt2lang_relabel_0607', 'w') as f:
    f.writelines(line for line in to_write)

import numpy as np
import random
def read_ivectors_0606(filepath, threshold):
  to_relabel = []
  utt2lang = {}
  with open('data/utt2lang_relabel_cleaned', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
  x_us = []
  x_uk = []
  x_nnm = []
  x_nnh = []
  files = os.listdir(filepath)
  for file in files:
    with open(f'{filepath}/{file}', 'r') as f:
      for line in f:
        item = line.strip().split()
        utt = item[0]
        try:
          label = utt2lang[utt]
        except KeyError:
          print('KeyError:', utt)
          continue
        vector = [float(x) for x in item[1:][1:-1]]
        if label == 'us':
          x_us.append((utt, vector))
        elif label == 'uk':
          x_uk.append((utt, vector))
        elif label == 'nnm':
          x_nnm.append((utt, vector))
        elif label == 'nnh':
          x_nnh.append((utt, vector))
  classes = ['us', 'uk', 'nnm', 'nnh']
  for cls in classes:
    samples = len(locals()[f'x_{cls}'])
    dist = {}
    # print('class', cls, samples)
    vectors = [x[1] for x in locals()[f'x_{cls}']]
    mean = get_mean(vectors, samples//10, 50)
    S = np.cov(np.array(vectors).T) #(600, 600)
    invS = np.linalg.inv(S)
    # if cls == 'nnh':
    #   print(invS)
    for x in locals()[f'x_{cls}']:
      utt = x[0]
      vector = x[1]
      dist[utt] = m_dist(vector, mean, invS)
    sorted_dist = {k: v for k, v in sorted(dist.items(), key=lambda item: item[1])}
    if cls == 'nnh':
      print(list(sorted_dist.items())[0], list(sorted_dist.items())[-1])
      for item in list(dist.items()):
        if item[1] > threshold:
          to_relabel.append(item[0] + ' ' + utt2lang[utt] + '\n')
    if cls == 'us':
      print(list(sorted_dist.items())[0], list(sorted_dist.items())[-1])
      for item in list(dist.items()):
        if item[1] > threshold:
          to_relabel.append(item[0] + ' ' + utt2lang[utt] + '\n')
    # if cls == 'nnh':
    #   for item in list(sorted_dist.items()):
    #     if not item[1] > 0:
    #       # print ('find')
    #       to_relabel.append(item[0] + ' ' + utt2lang[utt] + '\n')
    # else:
    #   for item in list(sorted_dist.items())[-samples//10:]:
    #     to_relabel.append(item[0] + ' ' + utt2lang[utt] + '\n')
  print(len(to_relabel))
  with open('utt2lang_2relabel', 'w') as f:
    f.writelines(line for line in to_relabel)

    # locals()[f'x_{cls}_mean'] = get_mean(locals()[f'x_{cls}'], samples//10, 50)
    # locals()[f'x_{cls}_S'] = np.cov(np.array(locals()[f'x_{cls}'].T))
    # locals()[f'x_{cls}_invS'] = np.linalg.inv(locals()[f'x_{cls}_S'])
    # print(locals()[f'x_{cls}_S'].shape)

def get_mean(x, k, n): # x:(samples, 600)
  all_means = []
  for _ in range(n):
    x_sample = random.sample(x, k)
    cur_mean = np.mean(np.array(x_sample), axis=0)
    all_means.append(cur_mean)
  x_mean = np.mean(all_means, axis=0)
  print('x_mean.shape:', x_mean.shape)
  return x_mean

def m_dist(x, y, invS): # x, y:(600,)
  tp = np.array(x-y)
  return np.sqrt(np.dot(np.dot(tp, invS), tp.T))


def find_valid():
  utt2lang = {}
  utt2link = {}
  utt2us = []
  utt2uk = []
  utt2nnl = []
  utt2nnn = []
  utt2nnm = []
  utt2nnh = []

  with open('data/utt2lang_relabel_0607', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang

  with open('utt2link', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      link = item[-1]
      utt2link[utt] = link

  with open(f"results/posteriors_2cls", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      try:
        lang = utt2lang[utt]
        link = utt2link[utt]
      except KeyError:
        print('KeyError:', utt)
        continue
      ok = float(item[2])
      nn = float(item[3])
      if lang == 'us':
        utt2us.append([utt, lang, ok, link])
      if lang == 'uk':
        utt2uk.append([utt, lang, ok, link])
      if lang == 'nnl':
        utt2nnl.append([utt, lang, ok, link])
      if lang == 'nnn':
        utt2nnn.append([utt, lang, ok, link])
      if lang == 'nnm':
        utt2nnm.append([utt, lang, nn, link])
      if lang == 'nnh':
        utt2nnh.append([utt, lang, nn, link])

  all = utt2us + utt2uk + utt2nnl + utt2nnn + utt2nnm + utt2nnh
  print(len(all))
  with open('all.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(line for line in all)

def difference():
  utt2lang_ours = {}
  utt2link = {}
  utt2lang_vivian = []
  with open('data/utt2lang_1710_2cls', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      lang = item[-1]
      utt2lang_ours[utt] = lang
  with open('data/utt2link_til0610', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      lang = item[-1]
      utt2link[utt] = lang
  with open('data/utt2lang_vivian', 'r') as f:
    for line in f:
      item = line.split()
      utt = item[0]
      lang = item[-1]
      if lang in ['uk', 'us', 'nnl', 'nnn']:
        lang = 'ok'
      else:
        lang = 'nn'
      if utt in utt2lang_ours.keys():
        if lang != utt2lang_ours[utt]:
          utt2lang_vivian.append([utt, utt2lang_ours[utt], lang, utt2link[utt]])
  print(len(utt2lang_vivian))
  # with open('difference', 'w') as f:
  # #   f.writelines(line for line in utt2lang_vivian)
  # with open('utt2lang_diff.csv', 'w') as f:
  #   writer = csv.writer(f, delimiter=',')
  #   writer.writerows(line for line in utt2lang_vivian)

def difference_error_124():
  utt_list124 = []
  error = []
  with open('error_0610_total124', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      utt_list124.append(utt)
  with open('error_446_2cls', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      gold = item[1]
      pred = item[2]
      if utt not in utt_list124:
        error.append([utt, gold, pred])
  print(len(error))
  with open('error_446_diff', 'w') as f:
    f.writelines(line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + '\n' for line in error)

def relabel0617():
  utt2lang = {}
  to_write = []
  with open(f"data/train_itg0610/utt2lang", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang
  with open(f"data/lre07_itg0610/utt2lang", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      utt2lang[utt] = lang

  tag2lang = {'non-native Heavy': 'nnh', 'non-native Maderate': 'nnm',
              'non-native light': 'nnl', 'neutral': 'nnn',
              'us': 'us', 'uk': 'uk'}
  workbook = xlrd.open_workbook('relabel/口音重新标注_iTG-0617.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    utt = str(sheet.row_values(i)[0])
    relabel = str(sheet.row_values(i)[4]).strip()
    if not relabel:
      del utt2lang[utt]
      continue
    relabel_ = tag2lang[relabel]
    utt2lang[utt] = relabel_

  for utt in utt2lang.keys():
    to_write.append([utt, utt2lang[utt]])
  print(len(to_write))
  with open('utt2lang_0610data_relabel_itg', 'w') as f:
    f.writelines(line[0] + ' ' + line[1] +  ' ' + '\n' for line in to_write)

def errorAfterRelabel_():
  utt2lang_pred = {}
  with open(f"posteriors_3and3_446_exp29", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nn = float(item[3])
      if nn > 0.7:
        utt2lang_pred[utt] = 'nn'
      else:
        utt2lang_pred[utt] = 'ok'

  cnt = 0
  with open(f"utt2lang_relabel_itg_446_6cls", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      relabel = item[-1]
      model = utt2lang_pred[utt]
      if relabel == 'uk' and model == 'nn':
        cnt += 1

  print(cnt)

def errorAfterRelabel():
  utt2lang_pred = {}
  with open(f"posteriors_3and3_446_exp29_nonnl", 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nn = float(item[3])
      if nn > 0.8:
        utt2lang_pred[utt] = 'nn'
      else:
        utt2lang_pred[utt] = 'ok'

  cnt = 0
  utt2lang = []
  utt2lang_relabel124 = {}
  tag2lang = {'Non-native Heavy': 'nnh', 'Non-native Moderate': 'nnm', 'Non-native moderate': 'nnm',
              'Non-native Light': 'nnl', 'Non-native Neutral': 'nnn',
              'Native American': 'us', 'native american': 'us', 'Native British': 'uk', 'Non-native light': 'nnl'}
  workbook = xlrd.open_workbook('relabel/重新标注_124.xlsx')
  # workbook = xlrd.open_workbook('relabel/relabel_error_0610_total124.xlsx')
  print(workbook.sheet_names())
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    utt = str(sheet.row_values(i)[0])
    # model = str(sheet.row_values(i)[2])
    model = utt2lang_pred[utt]
    relabel = str(sheet.row_values(i)[4]).strip()
    relabel_ = tag2lang[relabel]
    if relabel_ == 'nnl' and model == 'ok':
    # if relabel_ in ['nnm', 'nnh']:
      cnt += 1
  print(cnt)

    # # if relabel in ['us', 'uk', 'nnl', 'nnn']:
    # if relabel in ['Native American', 'Native British', 'Non-native Light ', 'Non-native Neutral']:
    #   relabel_ = 'ok'
    # else:
    #   relabel_ = 'nn'

    # utt2lang.append(utt + ' ' + relabel_ + '\n')
    # utt2lang_relabel124[utt] = relabel_
  #   # if model == relabel_:
  #     cnt += 1
  #   else:
  #     print(relabel, relabel_, model)
  # print(124-cnt)
  # print(len(utt2lang))

  # utt2lang_after_relabel = []
  # with open('utt2lang_446_6cls_before', 'r') as f:
  #   for line in f:
  #     item = line.split()
  #     utt = item[0]
  #     lang = item[-1]
  #     if utt in utt2lang_relabel124:
  #       lang = utt2lang_relabel124[utt]
  #     utt2lang_after_relabel.append(utt + ' ' + lang + '\n')
  # print(len(utt2lang_after_relabel))
  # with open('utt2lang_relabel_itg_446_6cls', 'w') as f:
  #   f.writelines(line for line in utt2lang_after_relabel)
  # with open('utt2lang_relabel_itg_124', 'w') as f:
  #   f.writelines(line for line in utt2lang)

def link_vname_utt():
  vname2utt = {}
  utt2lang_pred = []
  with open('/data/pytong/kaldi/egs/lre07/v1/utt2vname', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      vname = item[1]
      vname2utt[vname] = utt
  with open('0624_result.txt', 'r') as f:
    for line in f:
      item = line.strip().split()
      vname = item[0]
      lang = item[1]
      utt = vname2utt[vname]
      utt2lang_pred.append(utt + ' ' + lang + '\n')
  print(len(utt2lang_pred))
  with open('utt2lang180_pred', 'w') as f:
    f.writelines(line for line in utt2lang_pred)

def vname2lang():
  utt2vname = {}
  utt2lang_pred = []
  with open('/data/pytong/kaldi/egs/lre07/v1/utt2vname', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      vname = item[1]
      utt2vname[utt] = vname
  with open('/data/pytong/kaldi/egs/lre07/v1/exp_30/ivectors_itg_test180_exp30/output.2.0', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      vname = utt2vname[utt]
      utt2lang_pred.append(vname + ' ' + lang + '\n')
  print(len(utt2lang_pred))
  with open('testdata180_result2.0', 'w') as f:
    f.writelines(line for line in utt2lang_pred)

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

def test180_precision():
  vname2lang_paii = {}
  to_write = []
  workbook = xlrd.open_workbook('paii_test_result_0628.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    vname = str(sheet.row_values(i)[0])
    lang = str(sheet.row_values(i)[1])
    vname2lang_paii[vname] = lang
  print(len(vname2lang_paii))
  tag2lang = {'Non-native - Heavy': 'nnh', 'Non-native - Moderate': 'nnm',
              'Non-native - Light': 'nnl', 'Non-native - Neutral': 'nnn',
              'Native-American': 'us', 'Native-British': 'uk', 'None': 'nnl'}
  workbook = xlrd.open_workbook('iTG测试集人工标注.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[3])
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    spk = video_id.split('-')[0].split('_')[0]
    utt_id = f'{spk}-{answer_id}'
    tag = str(sheet.row_values(i)[1])
    try:
      paii_lang = vname2lang_paii[video_name]
    except KeyError as e:
      print('error,', utt_id)
    itg_lang = tag2lang[tag]
    itg_lang2 = 'nn' if itg_lang in ['nnm', 'nnh'] else 'ok'
    to_write.append([utt_id, original_link, paii_lang, itg_lang2, itg_lang])

  print(len(to_write))
  with open('test180_paii_itg.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(line for line in to_write)

def combine_post_ystar_ytuta(posteriors, output, utt2lang):
  utt2poster1 = {}
  utt2poster2 = {}
  utt2star = {}
  utt2pred = {}
  lang2tag = {'ok': 0, 'nn': 1}
  with open(utt2lang, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      tag = lang2tag[lang]
      utt2star[utt] = tag
  with open(output, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      tag = lang2tag[lang]
      utt2pred[utt] = tag
  with open(posteriors, 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      ok = float(item[2])
      nn = float(item[3])
      utt2poster1[utt] = ok
      utt2poster2[utt] = nn
  to_write = []
  for utt in utt2star.keys():
      to_write.append([utt, utt2poster1[utt], utt2poster2[utt], utt2pred[utt], utt2star[utt]])
  with open('combine_til0623.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(line for line in to_write)

def utt2lang_relabel_til0623():
  relabelfiles = ['utt2lang_0610data_relabel_itg', 'utt2lang3', 'utt2lang_relabel_itg_446_6cls']
  utt2lang_relabel = {}
  to_write = []
  for relabelfile in relabelfiles:
    with open(relabelfile, 'r') as f:
      for line in f:
        item = line.strip().split()
        utt = item[0]
        lang = item[-1]
        utt2lang_relabel[utt] = lang
  with open('utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if utt in utt2lang_relabel:
        to_write.append(utt + ' ' + utt2lang_relabel[utt] + '\n')
      else:
        to_write.append(utt + ' ' + lang + '\n')
  print(len(to_write))
  with open('utt2lang_relabel_til0623', 'w') as f:
    f.writelines(line for line in to_write)

import numpy as np
def count_conf_join(fPath):
  P = []
  file_list = []
  ids = []
  with open(fPath) as f:
    line = f.readline()
    while line:
      line = line.replace('\n', '')
      items = line.split('\t')
      print(items)
      ids.append(items[0])
      P.append([float(items[1]), float(items[2]), int(items[-1])])
      line = f.readline()
  m = len(P[0])
  n = len(P)
  C = [[0]*(m-1)for i in range(m-1)]
  t = Average(P)
  print(t)
  for i in range(n):
    cnt = 0
    y0 = -1
    p = 0
    for j in range(m-1):
      if P[i][j] >= t[j]:
        cnt+=1
        y0 = j if p <= P[i][j] else y0
        p = max(p, P[i][j])
    y1 = P[i][m-1]
    if str(P[i][-1]) == '0':
      lang = 'ok'
    else:
      lang = 'nn'

    if cnt > 0:
      C[y1][y0] += 1
      if y1 == y0: file_list.append(ids[i] + " " + lang)
  print(C)
  with open('utt2lang_filtered', 'w') as f:
    for line in file_list:
      line += '\n'
      f.write(line)
  return C

def Average(P):
  n  = len(P)
  m = len(P[0])
  t = [0]*(m-1)
  for j in range(m-1):
    l = []
    for i in range(n):
      if P[i][-1] == j:
        l.append(P[i][j])
    t[j] = sum(l)/len(l)
  return t

def mturk():
  to_write = []
  correct = 0
  workbook = xlrd.open_workbook('mturk/mturk_out1.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    count = {}
    video_id = str(sheet.row_values(i)[0])
    speaker_id = str(sheet.row_values(i)[0]).split('-')[0].split('_')[0]
    label1 = str(sheet.row_values(i)[1])
    label2 = str(sheet.row_values(i)[2])
    label3 = str(sheet.row_values(i)[3])
    itg_label = str(sheet.row_values(i)[-1])
    count[label1] = count.get(label1, 0) + 1
    count[label2] = count.get(label2, 0) + 1
    count[label3] = count.get(label3, 0) + 1
    try:
      mturk_label = [item[0] for item in count.items() if item[1] in [2, 3]][0]
    except IndexError:
      mturk_label = 'None'
    if mturk_label != 'None':
      correct += 1
    link = 'https://teacher-accent.s3-us-west-2.amazonaws.com/' + video_id
    to_write.append([speaker_id, label1, label2, label3, itg_label, mturk_label, link])
  print(correct)
  print(len(to_write))
  with open('mturk_out1_.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(line for line in to_write)

def correct_65():
  pred = {}
  count = 0
  with open('exp_35/ivectors_test_mturk_v3/output', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      pred[utt] = lang
  with open('exp_35/ivectors_test_mturk_v3/utt2lang_mturk_65', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[-1]
      if lang == pred[utt]:
        count+=1
      else:
        print(utt, lang, pred[utt])
  print(count)


if __name__ == '__main__':
  # dist = read_ivectors('results/v1/exp_23/ivectors.23')
  # print(dist[4])
  # read_ivectors_0606('results/ivectors.25.nnl', 35)
  # error_link()
  combine_post_ystar_ytuta('cl_test/til0623/posteriors', 'cl_test/til0623/output', 'cl_test/til0623/utt2lang')
  C = count_conf_join('combine_til0623.txt')
  print(C)

'''
https://stackoverflow.com/questions/29661574/normalize-numpy-array-columns-in-python
'''