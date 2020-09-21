import csv
import xlrd
from pa_nlp import *
from pa_nlp import nlp
import random
from collections import defaultdict
import json
map6 = {
  'Native American': 'us',
  'Native British': 'uk',
  'Non-native Heavy': 'nnh',
  'Non-native Moderate': 'nnm',
  'Native Neutral': 'nnn',
  'Non-native Light': 'nnl',
  'Others': 'Others'
}

map4_result = {
  'Native American': 'us',
  'Native British': 'uk',
  'Non-native': 'nna',
  'Neutral': 'nnn',
  'Others': 'Others'
}


map4 = {
  'us': 'us',
  'uk': 'uk',
  'nnh': 'nnh',
  'nnm': 'nnh',
  'nnn': 'nnl',
  'nnl': 'nnl',
  'Others': 'Others'
}

map4_2 = {
  'us': 'us',
  'uk': 'uk',
  'nna': 'nna',
  'nnn': 'nnn',
  'Others': 'Others'
}

label2num = {
  'us': 0,
  'uk': 1,
  'nnn': 2,
  'nna': 3,
  'Others': 10
}

def filter(csv_path, out_path, utt2lang, out_utt2lang6, out_utt2lang4,
           distinct_utt2lang6, distinct_utt2lang4, itg_to_write):
  content = {}
  old_label = {}

  # read itg label
  with open(utt2lang) as f:
    line = f.readline()
    while line:
      line = line.replace('\n', '')
      items = line.split()
      id = items[0].split('-')[0]
      old_label[id] = [items[1], items[0]]
      line = f.readline()

  c6, c4 = 0, 0 # mturk label is consistent
  match6, match4 = 0, 0 # mturk label match itg label

  # read mturk csv
  with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
      if line_count == 0:
        line_count += 1
      else:
        wav_name = row[-2].split('/')[-1]
        label = row[-1]
        label = label.split(':')[0].split('：')[0].strip()
        label = map6.get(label)
        if wav_name not in content:
          id = wav_name.split('-')[0].split('_')[0]
          try:
            content[wav_name] = [old_label[id][1], row[-2]]
          except KeyError:
            continue
        content[wav_name].append(label)

  # merge mturk label and itg labels
  for wav_name in content:
    id = wav_name.split('-')[0].split('_')[0] # EE
    count6 = {}
    count4 = {}
    max_label6 = None
    max_label4 = None
    for label6 in content[wav_name][2:]:
      label4 = map4.get(label6)
      count6[label6] = count6.get(label6, 0) + 1
      count4[label4] = count4.get(label4, 0) + 1
      if count6[label6] > 1: max_label6 = label6
      if count4[label4] > 1: max_label4 = label4
    if max_label6 is not None and id not in distinct_utt2lang6:
      c6 += 1
      out_utt2lang6.append(old_label.get(id)[1] + ' ' + max_label6 + '\n')
      itg_to_write.append(old_label.get(id)[1] + ' ' + old_label.get(id)[0] + '\n')
      distinct_utt2lang6.append(id)

    if max_label4 is not None and id not in distinct_utt2lang4:
      c4 += 1
      out_utt2lang4.append(old_label.get(id)[1] + ' ' + max_label4 + '\n')
      distinct_utt2lang4.append(id)

    content[wav_name].append(max_label6)
    content[wav_name].append(max_label4)

    if id in old_label:
      v = old_label.get(id)[0]
      if max_label6 == v:
        match6 += 1

      if max_label4 == v:
        match4 += 1

      content[wav_name].append(v)

  print(csv_path, str(c6) + " " + str(c4) + " " + str(match6) + " " + str(match4))

  # to_write = list(content.values())
  # heads = ['id', 'url', 'label1', 'label2', 'label3', 'mturk_label6', 'mturk_label4', 'itg_label']
  # to_write.insert(0, heads)
  # with open(out_path, 'w', newline='') as file:
  #   writer = csv.writer(file)
  #   writer.writerows(to_write)
  return c6, c4, match6, match4

def mturk_nn_filter():
  unique_answers = []
  urls = []
  tag2lang = {'Non-native - Heavy': 'nnh', 'Non-native - Moderate': 'nnm',
              'Non-native - Light': 'nnl', 'Non-native - Neutral': 'nnn',
              'Native-American': 'us', 'Native-British': 'uk'}
  workbook = xlrd.open_workbook('data.info/iTG_0623.xlsx')
  sheet = workbook.sheet_by_name('accent')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[6]).replace(' ', '%20')
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    tag = str(sheet.row_values(i)[1])
    if answer_id in unique_answers:
      continue
    try:
      lang = tag2lang[tag]
    except KeyError:
      continue
    if lang not in ['us', 'uk']:
      urls.append(['https://teacher-accent.s3-us-west-2.amazonaws.com/'+video_id+'.wav'])

  workbook = xlrd.open_workbook('data.info/iTG_0703.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[5]).replace(' ', '%20')
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    tag = str(sheet.row_values(i)[1])
    if answer_id in unique_answers:
      continue
    try:
      lang = tag2lang[tag]
    except KeyError:
      continue
    if lang not in ['us', 'uk']:
      urls.append(['https://teacher-accent.s3-us-west-2.amazonaws.com/'+video_id+'.wav'])

  workbook = xlrd.open_workbook('data.info/iTG_0716.xlsx')
  sheet = workbook.sheet_by_name('Sheet1')
  for i in range(1, sheet.nrows):
    answer_id = str(sheet.row_values(i)[0])
    original_link = str(sheet.row_values(i)[5]).replace(' ', '%20')
    video_name = original_link.split('/')[-1]
    video_id = video_name.split('.')[0]
    tag = str(sheet.row_values(i)[1])
    if answer_id in unique_answers:
      continue
    try:
      lang = tag2lang[tag]
    except KeyError:
      continue
    if lang not in ['us', 'uk']:
      urls.append(['https://teacher-accent.s3-us-west-2.amazonaws.com/'+video_id+'.wav'])

  print(len(urls))
  fields = ['audio_url']
  filename = "mturk/nn_csv/s3url"
  branchs = int(len(urls)/50 + 1)
  print(branchs)
  for i in range(branchs):
    suburl = urls[i*50:min(len(urls), (i+1)*50)]
    if i == 0: print(len(suburl))
    with open(filename+str(i)+'.csv', 'w') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(fields)
      csvwriter.writerows(suburl)

def mturk_relabel_2round():
  utt_list = []
  urls = []
  utt2itglang = {}
  with open('data/utt2lang_til0716', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nid = item[-1]
      utt2itglang[utt] = nid
  with open('mturk/utt2lang6.v6', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      utt_list.append(utt)
  with open('data/wav.scp.til0716', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      if utt in utt_list:
        continue
      if utt2itglang[utt] in ['us', 'uk']:
        continue
      link = item[-1]
      video_name = link.split('/')[-1]
      url = 'https://teacher-accent.s3-us-west-2.amazonaws.com/' + video_name
      urls.append([url])
  print(len(urls))
  fields = ['audio_url']
  filename = "mturk/2nd/s3url"
  branchs = int(len(urls)/500 + 1)
  print(branchs)
  for i in range(branchs):
    suburl = urls[i*500:min(len(urls), (i+1)*500)]
    if i == 0: print(len(suburl))
    with open(filename+str(i)+'.csv', 'w') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(fields)
      csvwriter.writerows(suburl)

def mturk_relabel_newdata_nn():
  urls = []
  utt2itglang = {}
  with open('data/data_itg0826/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      nid = item[-1]
      utt2itglang[utt] = nid
  with open('data/data_itg0826/wav.scp', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      if utt2itglang[utt] in ['us', 'uk']:
        continue
      link = item[-1]
      video_name = link.split('/')[-1]
      url = 'https://teacher-accent.s3-us-west-2.amazonaws.com/' + video_name
      urls.append([url])
  print(len(urls))
  fields = ['audio_url']
  filename = "mturk/s3url/0826_nn/s3url"
  branchs = int(len(urls)/500 + 1)
  print(branchs)
  for i in range(branchs):
    suburl = urls[i*500:min(len(urls), (i+1)*500)]
    if i == 0: print(len(suburl))
    with open(filename+str(i)+'.csv', 'w') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(fields)
      csvwriter.writerows(suburl)


def mturk_data_resample(input_data, wav_scp, output_file, output_list=None):
  data = list(nlp.pydict_file_read(input_data))
  wav_path_dict = dict()
  wav_path = [item.strip().split() for item in open(wav_scp, 'r').readlines()]

  for wav in wav_path:
    if wav[0] not in wav_path_dict:
      wav_path_dict[wav[0]] = wav[1]

  class_dict = defaultdict(list)
  sample_list = []
  audio_samples = defaultdict()
  for d in data:
    # if d.get('class') not in class_dict:
    #   class_dict[d.get('class')].append(d.get('id'))
    class_dict[d.get('class')].append(d.get('id'))
  accent_list = class_dict[3] + class_dict[4] +class_dict[5]
  print(f'There are {len(class_dict[0])} us samples')
  print(f'There are {len(class_dict[1])} uk samples')
  print(f'There are {len(class_dict[2])} neutral samples')
  print(f'There are {len(class_dict[3])} nn light samples')
  print(f'There are {len(class_dict[4])} nn moderate samples')
  print(f'There are {len(class_dict[5])} nn heavy samples')
  print(f'There are {len(accent_list)} nn samples')
  for k, v in class_dict.items():
    if k in [0, 1, 2]:
      sample_list += random.sample(v, 125)
    if isinstance(k, int):

      audio_samples[k] = dict(utt_id=random.sample(v, 5), wav_path=[])

  sample_list += random.sample(accent_list, 125)
  # audio_samples = random.sample(accent_list, 5)
  for k, v in audio_samples.items():
    for utt in v['utt_id']:
      audio_samples[k]['wav_path'].append(wav_path_dict[utt])
    # v.update({'wav_path': wav_path_dict[k]})
  if output_list:
    with open(output_list, 'w')as f_list:
      json.dump(audio_samples, f_list, indent=2)
  urls = []
  for sample in sample_list:
    video_name = wav_path_dict[sample].split('/')[-1]
    url = 'https://teacher-accent.s3-us-west-2.amazonaws.com/' + video_name
    urls.append([url])
  fields = ['audio_url']
  with open(output_file, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(urls)



def filter2(csv_path):
  content = {}
  old_label = {}

  # read itg label
  # with open(utt2lang) as f:
  #   line = f.readline()
  #   while line:
  #     line = line.replace('\n', '')
  #     items = line.split()
  #     id = items[0].split('-')[0]
  #     old_label[id] = [items[1], items[0]]
  #     line = f.readline()

  c6, c4 = 0, 0 # mturk label is consistent
  # match6, match4 = 0, 0 # mturk label match itg label

  # read mturk csv
  with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
      if line_count == 0:
        line_count += 1
      else:
        wav_name = row[-2].split('/')[-1]
        label = row[-1]
        label = label.split(':')[0].split('：')[0].strip()
        label = map4_result.get(label)
        if wav_name not in content:
          id = wav_name.split('-')[0].split('_')[0]
          try:
            content[wav_name] = [row[-2]]
          except KeyError:
            continue
        content[wav_name].append(label)

  # merge mturk label and itg labels
  for wav_name in content:
    id = wav_name.split('-')[0].split('_')[0] # EE
    count6 = {}
    count4 = {}
    max_label6 = None
    max_label4 = None
    for label6 in content[wav_name][1:]:
      label4 = map4_2.get(label6)
      count6[label6] = count6.get(label6, 0) + 1
      count4[label4] = count4.get(label4, 0) + 1
      if count6[label6] > 1: max_label6 = label6
      if count4[label4] > 1: max_label4 = label4
    if max_label6 is not None:
      c6 += 1
    #   out_utt2lang6.append(old_label.get(id)[1] + ' ' + max_label6 + '\n')
    #   itg_to_write.append(old_label.get(id)[1] + ' ' + old_label.get(id)[0] + '\n')
    #   distinct_utt2lang6.append(id)
    #
    if max_label4 is not None:
      c4 += 1
    #   out_utt2lang4.append(old_label.get(id)[1] + ' ' + max_label4 + '\n')
    #   distinct_utt2lang4.append(id)

    content[wav_name].append(max_label6)
    content[wav_name].append(max_label4)

    # if id in old_label:
    #   v = old_label.get(id)[0]
    #   if max_label6 == v:
    #     match6 += 1
    #
    #   if max_label4 == v:
    #     match4 += 1
    #
    #   content[wav_name].append(v)

  # print(csv_path, str(c6) + " " + str(c4) + " " + str(match6) + " " + str(match4))
  print(csv_path,
        str(c6) + " " + str(c4))
  # to_write = list(content.values())
  # heads = ['id', 'url', 'label1', 'label2', 'label3', 'mturk_label6', 'mturk_label4', 'itg_label']
  # to_write.insert(0, heads)
  # with open(out_path, 'w', newline='') as file:
  #   writer = csv.writer(file)
  #   writer.writerows(to_write)
  return c6, c4

def filter3(csv1, csv2):
  content = {}
  old_label = {}
  first_label = dict()
  total_result = defaultdict()
  #first batch mturk label:
  c6_1,c4_1 = 0, 0
  with open(csv1) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
      if line_count == 0:
        line_count += 1
      else:
        wav_name = row[-2].split('/')[-1]
        label = row[-1]
        label = label.split(':')[0].split('：')[0].strip()
        label = map4_result.get(label)
        if wav_name not in first_label:
          id = wav_name.split('-')[0].split('_')[0]
          try:
            first_label[wav_name] = [row[-2]]
          except KeyError:
            continue
        first_label[wav_name].append(label)

  for wav_name in first_label:
    id = wav_name.split('-')[0].split('_')[0]  # EE
    count6_1 = {}
    count4_1 = {}
    max_label6 = None
    max_label4 = None
    for label6 in first_label[wav_name][1:]:
      label4 = map4_2.get(label6)
      count6_1[label6] = count6_1.get(label6, 0) + 1
      count4_1[label4] = count4_1.get(label4, 0) + 1
      if count6_1[label6] > 1: max_label6 = label6
      if count4_1[label4] > 1: max_label4 = label4
    if max_label6 is not None:
      c6_1 += 1

    if max_label4 is not None:
      c4_1 += 1

    first_label[wav_name].append(max_label6)
    first_label[wav_name].append(max_label4)


    if wav_name not in total_result and max_label6 is not None:
      total_result[wav_name] = [label2num[max_label6]]
    else:
      total_result[wav_name] = ['100']


  # read itg label
  # with open(utt2lang) as f:
  #   line = f.readline()
  #   while line:
  #     line = line.replace('\n', '')
  #     items = line.split()
  #     id = items[0].split('-')[0]
  #     old_label[id] = [items[1], items[0]]
  #     line = f.readline()

  c6, c4 = 0, 0  # mturk label is consistent
  match6, match4 = 0, 0 # mturk label match itg label (or other batch of mturk labelling)

  # read mturk csv
  with open(csv2) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
      if line_count == 0:
        line_count += 1
      else:
        wav_name = row[-2].split('/')[-1]
        label = row[-1]
        label = label.split(':')[0].split('：')[0].strip()
        label = map4_result.get(label)
        if wav_name not in content:
          id = wav_name.split('-')[0].split('_')[0]
          try:
            content[wav_name] = [row[-2]]
          except KeyError:
            continue
        content[wav_name].append(label)

  # merge mturk label and itg labels
  for wav_name in content:
    id = wav_name.split('-')[0].split('_')[0]  # EE
    count6 = {}
    count4 = {}
    max_label6 = None
    max_label4 = None
    for label6 in content[wav_name][1:]:
      label4 = map4_2.get(label6)
      count6[label6] = count6.get(label6, 0) + 1
      count4[label4] = count4.get(label4, 0) + 1
      if count6[label6] > 1: max_label6 = label6
      if count4[label4] > 1: max_label4 = label4

    if max_label6 is not None:
      c6 += 1
    #   out_utt2lang6.append(old_label.get(id)[1] + ' ' + max_label6 + '\n')
    #   itg_to_write.append(old_label.get(id)[1] + ' ' + old_label.get(id)[0] + '\n')
    #   distinct_utt2lang6.append(id)
    #
    if max_label4 is not None:
      c4 += 1
    #   out_utt2lang4.append(old_label.get(id)[1] + ' ' + max_label4 + '\n')
    #   distinct_utt2lang4.append(id)

    content[wav_name].append(max_label6)
    content[wav_name].append(max_label4)
    if max_label6 is not None:
      total_result[wav_name].append(label2num[max_label6])
    else:
      total_result[wav_name].append('100')

    if wav_name in first_label:
      v = first_label.get(wav_name)[-1]
      if max_label6 == v:
        match6 += 1

      if max_label4 == v:
        match4 += 1

      content[wav_name].append(v)


  # print(csv_path, str(c6) + " " + str(c4) + " " + str(match6) + " " + str(match4))
  print(csv1, csv2,
        str(match4) + " " + str(match6))
  # to_write = list(content.values())
  # heads = ['id', 'url', 'label1', 'label2', 'label3', 'mturk_label6', 'mturk_label4', 'itg_label']
  # to_write.insert(0, heads)
  # with open(out_path, 'w', newline='') as file:
  #   writer = csv.writer(file)
  #   writer.writerows(to_write)
  return total_result

def gen_predict_text(data, output_text):
  with open (output_text, 'w') as f_o:
    for k, v in data.items():
      if int(v[0]) !=100 and int(v[1]) !=100:
        print(v[0], v[1], file=f_o)

def filter_same_label(data, output_same_label):
  c_0, c_1, c_2, c_3,  = 0, 0, 0, 0,
  with open(output_same_label, 'w') as f_o:

    for k, v in data.items():

      if v[0] != 100 and v[1] != 100 :
        if v[0] == v[1]:

          print(v[0], v[1], file=f_o)

if __name__ == "__main__":
  # mturk_relabel_newdata_nn()
  ####original generate label result from beginning or the recent result
  # total6, total4, match_6, match_4 = 0, 0, 0, 0
  # utt2lang6, utt2lang4 = [], []
  # distinct_utt2lang6 = []
  # distinct_utt2lang4 = []
  # itg_to_write = []
  # for i in range(144, 145):
  #   c6, c4, match6, match4 = filter(f'mturk/original/{i}.csv',
  #                                   f'mturk/out/out{i}.csv',
  #                                   'data/data_itg0826/utt2lang',
  #                                   utt2lang6, utt2lang4,
  #                                   distinct_utt2lang6, distinct_utt2lang4,
  #                                   itg_to_write)
  #   total6 += c6
  #   total4 += c4
  #   match_6 += match6
  #   match_4 += match4
  # print(total6, total4)
  # print(match_6, match_4)
  ##############################################################
  # with open('mturk/utt2lang6.v9', 'w') as f:
  #   f.writelines(line for line in utt2lang6)
  # with open('mturk/utt2lang4.v9', 'w') as f:
  #   f.writelines(line for line in utt2lang4)
  # print(len(itg_to_write))
  # with open('mturk/itg_label.v9', 'w') as f:
  #   f.writelines(line for line in itg_to_write)
##################generate samples fro mturk labelling ####################################
  # input_data = 'data/data_til0819/data.03.train.pydict'
  # wav_scp = 'data/data_til0819/wav.scp'
  # output_csv = 'mturk/sample_500/sample_500.csv'
  # audio_samples = 'mturk/sample_500/sample_5.json'
  # mturk_data_resample(input_data, wav_scp, output_csv, output_list=audio_samples)

###############analysis 4 categories label results ##########################
  batch1_csv_path = 'mturk/original/sample_500_batch1_results.csv'
  batch2_csv_path = 'mturk/original/sample_500_batch2_results.csv'
  output_txt = 'codes/prediction_result2.txt'
  output_same_label = 'codes/same_results.txt'
  res = filter3(batch1_csv_path, batch2_csv_path)
  gen_predict_text(res, output_txt)
  # filter2(batch1_csv_path)
  filter_same_label(res, output_same_label)
  c_0, c_1, c_2, c_3,  = 0, 0, 0, 0
  for k, v in res.items():
    if v[0] == v[1]:
      if int(v[0]) == 0:
        c_0 +=1
      elif int(v[0]) == 1:
        c_1 +=1
      elif int(v[0]) == 2:
        c_2 +=1
      elif int(v[0]) == 3:
        c_3 +=1
  print(f'0 is {c_0}')
  print(f'1 is {c_1}')
  print(f'2 is {c_2}')
  print(f'3 is {c_3}')