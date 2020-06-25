
import random
import xlrd
import re
import os

def listdir_nohidden(path):
  res = [f for f in os.listdir(path) if not f.startswith('.')]
  return res

def download():
  basepath = os.path.expanduser('~/Downloads/youtube')
  workbook = xlrd.open_workbook('youtube.xlsx')
  sheet = workbook.sheet_by_name('metadata')
  reg_1 = r'watch\?v=(.*)&list'
  reg_2 = r'watch\?v=(.*)'
  prefix = 'https://www.youtube.com/watch?v='

  for i in range(1, sheet.nrows):
    if sheet.row_values(i)[3]: # if label
      name = str(sheet.row_values(i)[1]) # sheet.cell_value(i,1)
      original_link = str(sheet.row_values(i)[4])

      source = str(sheet.row_values(i)[5]).lower()
      if not os.path.exists(f'{basepath}/{source}'):
        os.system(f'mkdir {basepath}/{source}')

      try:
        video_id = re.findall(reg_1, original_link)[0]
      except IndexError:
        try:
          video_id = re.findall(reg_2, original_link)[0]
        except IndexError:
          video_id = original_link

      if not os.path.exists(f'{basepath}/{source}/{video_id}.mp3'):
        download_link = prefix + video_id
        os.chdir(f'{basepath}/{source}')
        os.system(f'youtube-dl -x --audio-format mp3 -o {video_id}.1 {download_link}')

def do_segments():
  basepath = os.path.expanduser('~/Downloads/youtube')
  # sources = [f for f in os.listdir(basepath) if not f.startswith('.')]
  sources = ['oxford', 'others', 'british_english_accent_speech', 'tedxlondon',
             'american_english_accent_speech', 'tedxnewyork']
  for source in sources:
    files = [file for file in os.listdir(f'{basepath}/{source}')
             if file.split('.')[-1]=='wav']
    for file in files:
      if not os.path.exists(f"{basepath}/{source}/{file.split('.')[0]}"):
        os.system(f"mkdir {basepath}/{source}/{file.split('.')[0]}")
      os.system(f"ffmpeg -i {basepath}/{source}/{file} "
                f"-ss 30 -f segment -segment_time 30 -c copy "
                f"{basepath}/{source}/{file.split('.')[0]}/{file.split('.')[0]}_%03d.wav")

def convert_format():
  basepath = os.path.expanduser('~/Downloads/youtube')
  sources = ['oxford', 'others', 'british_english_accent_speech', 'tedxlondon',
             'american_english_accent_speech', 'tedxnewyork']
  for source in sources:
    files = os.listdir(f'{basepath}/{source}')
    for file in files:
      os.system(f"ffmpeg -i {basepath}/{source}/{file} "
                f"-vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav "
                f"{basepath}/{source}/{file.split('.')[0]}.wav")

def prepare_data():
  basepath = os.path.expanduser('~/Downloads/youtube')
  workbook = xlrd.open_workbook('youtube.xlsx')
  sheet = workbook.sheet_by_name('metadata')
  recording_ids = [id for id in sheet.col_values(0)[1:] if id]
  dataset = 'train'

  random.seed(0)
  test = random.sample(recording_ids, int((len(recording_ids))/10))

  reg_1 = r'watch\?v=(.*)&list'
  reg_2 = r'watch\?v=(.*)'

  utt2wav = []
  utt2lang = []
  utt2spk = []
  segments = []

  for i in range(1, sheet.nrows):
    if sheet.row_values(i)[3]: # if label
      id = str(sheet.row_values(i)[0])
      if id in test:
        continue
      name = str(sheet.row_values(i)[1].replace(' ', ''))
      name = str(name.replace('-', ''))
      lang = str(sheet.row_values(i)[3])
      original_link = str(sheet.row_values(i)[4])
      source = str(sheet.row_values(i)[5]).lower()

      try:
        video_id = re.findall(reg_1, original_link)[0]
      except IndexError:
        try:
          video_id = re.findall(reg_2, original_link)[0]
        except IndexError:
          video_id = original_link

      audios = listdir_nohidden(f"{basepath}/{source}/{video_id}")
      numbers = [30*(int(audio.split('_')[-1].split('.')[0])+1) for audio in audios]
      starts = ['0'*(4-len(str(number)))+str(number)+'00' for number in numbers]
      ends = ['0'*(4-len(str(number+30)))+str(number+30)+'00' for number in numbers]
      utterances = [f"{name}-{id}-{start}-{end}" for start, end in zip(starts, ends)]

      segments.extend([utt + ' ' + id + ' ' + str(float(number)) + ' '
                       + str(float(number+30)) + '\n'
                       for number, utt in zip(numbers, utterances)])
      utt2lang.extend([utt + ' ' + lang + '\n' for utt in utterances])
      utt2spk.extend([utt + ' ' + name + '\n' for utt in utterances])
      utt2wav.append(id + ' ' +
                     f"/data/pytong/wav/youtube/{source}/{video_id}.wav" + '\n')

  with open(f'data/{dataset}_tmp/segments', 'w') as f:
    f.writelines(line for line in segments)

  with open(f'data/{dataset}_tmp/wav.scp', 'w') as f:
    f.writelines(line for line in utt2wav)

  with open(f'data/{dataset}_tmp/utt2lang', 'w') as f:
    f.writelines(line for line in utt2lang)

  with open(f'data/{dataset}_tmp/utt2spk', 'w') as f:
    f.writelines(line for line in utt2spk)


def spk2gender():
  spk2gender = []
  unique_spks = []

  workbook = xlrd.open_workbook('youtube.xlsx')
  sheet = workbook.sheet_by_name('metadata')

  recording_ids = [id for id in sheet.col_values(0)[1:] if id]
  dataset = 'train'

  random.seed(0)
  test = random.sample(recording_ids, int((len(recording_ids))/10))

  for i in range(1, sheet.nrows):
    if sheet.row_values(i)[3]: # if label
      id = str(sheet.row_values(i)[0])
      if id in test:
        continue
      name = str(sheet.row_values(i)[1].replace(' ', ''))
      name = str(name.replace('-', ''))
      if name in unique_spks:
        continue
      unique_spks.append(name)
      gender = str(sheet.row_values(i)[2])
      spk2gender.append(name + ' ' + gender + '\n')

  with open(f'data/{dataset}_youtube/spk2gender', 'w') as f:
    f.writelines(line for line in spk2gender)


if __name__ == '__main__':
  prepare_data()