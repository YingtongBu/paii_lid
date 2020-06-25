import os
import sys
import random
import numpy as np

'''client_id       path    sentence        up_votes        down_votes      age     gender  accent
    1a0446d48b2fe1114209fcb872c1ca88b6eb43d420aeb81255ee4ff9d6e295c56257b7cf2ff2dde3905f1675a0a29c728187cc03b0423d68ddf7fb1539e8f399
    common_voice_en_17250490.mp3 "I'd like to order some cinnamon pretzels, please."     1       0       fourties        male    newzealand'''

'''['us', 'england', 'african', 'indian', 'other', 'hongkong', 
'philippines', 'canada', 'singapore', 'scotland', 'australia', 'bermuda', 
'newzealand', 'ireland', 'malaysia', 'wales', 'southatlandtic']'''
def unique_langs():
  unique_langs = []
  with open('tsv/validated.tsv', 'r') as f:
    next(f)
    for line in f:
      item = line.strip().split('\t')
      if len(item) < 8:
        continue
      lang = item[-1]
      if lang in unique_langs:
        continue
      unique_langs.append(lang)
  return unique_langs

def convert_format(i):
  file_name = f"v0{i}"
  with open(f"tsv/{file_name}", 'r') as f:
    for line in f:
      item = line.strip().split('\t')
      if len(item) < 8:
        continue
      lang = item[-1]
      if lang == 'other':
        continue
      recording_id = item[1].split('.')[0]
      try:
        os.system(f"ffmpeg -i clips/{recording_id}.mp3 "
                  f"-vn -acodec pcm_s16le -ac 1 -ar 16000 -f wav "
                  f"lid_wav/{recording_id}.wav")
      except:
        print(f"Error! passed: {recording_id}")

'''404506'''
def prepare_data(i):
  unique_spks = []
  utt2wav = []
  utt2lang = []
  utt2spk = []
  spk2gender = []
  with open(f'tsv/v{i:02d}', 'r') as f:
    for line in f:
      item = line.strip().split('\t')
      if len(item) < 8:
        continue
      lang = item[-1]
      if lang == 'other':
        continue
      recording_id = item[1].split('.')[0]
      if f"{recording_id}.wav" not in wavs:
        continue
      if lang == 'england':
        lang = 'uk'
      spk_id = item[0]
      path = f"/data/pytong/wav/comvoc/lid_wav/{recording_id}.wav"
      utt = f"{spk_id}-{recording_id}"
      print(utt)
      utt2wav.append(utt + ' ' + path + '\n')
      utt2lang.append(utt + ' ' + lang + '\n')
      utt2spk.append(utt + ' ' + spk_id + '\n')
      gender = 'm' if item[6] == 'male' else 'f'
      if spk_id in unique_spks:
        continue
      unique_spks.append(spk_id)
      spk2gender.append(spk_id + ' ' + gender + '\n')
  with open(f'data/spk2gender', 'a') as f:
    f.writelines(line for line in spk2gender)
  with open(f'data/wav.scp', 'a') as f:
    f.writelines(line for line in utt2wav)
  with open(f'data/utt2lang', 'a') as f:
    f.writelines(line for line in utt2lang)
  with open(f'data/utt2spk', 'a') as f:
    f.writelines(line for line in utt2spk)
    # spk2gender_ = []
    # with open(f'data/{dataset}/utt2lang', 'a') as f:
    #   for line in f:
    #     item = line.strip().split()
    #     spk_id = item[0].split('-')[0]
    #     if spk_id in unique_spks:
    #       continue
    #     unique_spks.append(spk_id)
    #     spk2gender_.append(spk_id + ' ' + spk2gender[spk_id] + '\n')
    # with open(f'data/{dataset}/spk2gender', 'w') as f:
    #   f.writelines(line for line in spk2gender_)


def stat():
  utt_by_lang = dict()
  spk_by_lang = dict()
  male_by_lang = dict()
  female_by_lang = dict()
  lang2spk = dict()
  spk2gender = dict()
  with open(f'data/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      lang = item[-1]
      spk = item[0].split('-')[0]
      if lang not in utt_by_lang:
        utt_by_lang[lang] = 1
      else:
        utt_by_lang[lang] += 1
      if lang not in lang2spk:
        lang2spk[lang] = []
      if spk not in lang2spk[lang]:
        lang2spk[lang].append(spk)
  with open(f'data/spk2gender', 'r') as f:
    for line in f:
      item = line.strip().split()
      gender = item[-1]
      spk = item[0]
      spk2gender[spk] = gender
  lang2mspk = dict()
  lang2fspk = dict()
  for lang in lang2spk.keys():
    lang2mspk[lang] = []
    lang2fspk[lang] = []
    for spk in lang2spk[lang]:
      if spk2gender[spk] == 'm':
        lang2mspk[lang].append(spk)
      elif spk2gender[spk] == 'f':
        lang2fspk[lang].append(spk)
  for lang in lang2spk.keys():
    spk_by_lang[lang] = len(lang2spk[lang])
    male_by_lang[lang] = len(lang2mspk[lang])
    female_by_lang[lang] = len(lang2fspk[lang])
  return utt_by_lang, spk_by_lang, male_by_lang, female_by_lang

def filter_nn():
  utt2lang = []
  with open('utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      spk = utt.split('-')[0]
      lang = item[1]
      if lang == 'african':
        lang = 'africa'
      elif lang == 'indian':
        lang = 'india'
      if lang in ['africa', 'india', 'hongkong', 'singapore', 'philippines',
                  'australia', 'bermuda', 'newzealand', 'malaysia', 'uk', 'us']:
        utt2lang.append(utt + ' ' + lang + '\n')
      if spk == '71478fb513ad412528972ba908369d72ba70339b171e24968cfe5a3421ac6959e945edc4b5f0fc952bda9bcbe6895518c33398159323093f9287e4898cf702a5':
        utt2lang.append(utt + ' ' + lang + '\n')
  with open('utt2lang.filtered', 'w') as f:
    f.writelines(line for line in utt2lang)


def three_classes():
  utt2lang = []
  with open('utt2lang.6', 'r') as f:
    for line in f:
      item = line.strip().split()
      utt = item[0]
      lang = item[1]
      if lang not in ['uk', 'us']:
        lang = 'nn'
      utt2lang.append(utt + ' ' + lang + '\n')
  with open(f'utt2lang.3', 'w') as f:
    f.writelines(line for line in utt2lang)

def split_india():
  utt2lang_dict = {}
  unique_spks = []
  utt2wav = []
  utt2spk = []
  utt2lang = []
  utt2wav_test = []
  utt2spk_test = []
  utt2lang_test = []
  with open('comvoc_filtered/utt2lang', 'r') as f:
    for line in f:
      item = line.strip().split()
      lang = item[-1]
      utt = item[0]
      if lang != 'india':
        continue
      utt2lang_dict[utt] = lang
      spk = utt.split('-')[0]
      if spk not in unique_spks:
        unique_spks.append(spk)
  random.seed(0)
  test_spks_indices = sorted(random.sample(range(len(unique_spks)), int((len(unique_spks))/10)))
  test_spks = np.array(unique_spks)[test_spks_indices]
  with open('comvoc_filtered/wav.scp', 'r') as f:
    for line in f:
      item = line.strip().split()
      dir = item[-1]
      utt = item[0]
      try:
        lang = utt2lang_dict[utt]
      except KeyError:
        continue
      spk = utt.split('-')[0]
      if spk in test_spks:
        utt2lang_test.append(utt + ' ' + lang + '\n')
        utt2spk_test.append(utt + ' ' + spk + '\n')
        utt2wav_test.append(utt + ' ' + dir + '\n')
      else:
        utt2lang.append(utt + ' ' + lang + '\n')
        utt2spk.append(utt + ' ' + spk + '\n')
        utt2wav.append(utt + ' ' + dir + '\n')
  with open('train_comvoc_india/wav.scp', 'w') as f:
    f.writelines(line for line in utt2wav)
  with open('train_comvoc_india/utt2lang', 'w') as f:
    f.writelines(line for line in utt2lang)
  with open('train_comvoc_india/utt2spk', 'w') as f:
    f.writelines(line for line in utt2spk)
  with open('lre07_comvoc_india/wav.scp', 'w') as f:
    f.writelines(line for line in utt2wav_test)
  with open('lre07_comvoc_india/utt2lang', 'w') as f:
    f.writelines(line for line in utt2lang_test)
  with open('lre07_comvoc_india/utt2spk', 'w') as f:
    f.writelines(line for line in utt2spk_test)


if __name__ == '__main__':
  wavs = [wav for wav in os.listdir('/data/pytong/wav/comvoc/lid_wav') if wav.endswith('.wav')]
  # i = int(sys.argv[1])
  # prepare_data(i)
  # # convert_format(i) '''should be 415189 clips in wav format'''
  #
  # utt_by_lang, spk_by_lang, male_by_lang, female_by_lang = stat()


'''
for ((i=0;i<=42;i++));do (nohup python prepare_data.py $i > log/write.log/write.log$i &)& done
'''