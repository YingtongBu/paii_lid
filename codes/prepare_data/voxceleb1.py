
import os

def prepare_data():
  spk2lang = dict()
  spk2gender = dict()
  spk2dataset = dict()


  with open('VoxCeleb1.speaker.info', 'r') as f:
    for line in f:
      item = line.strip().split()
      spk_id = item[0]
      gender = item[2]
      lang = item[3].lower()
      dataset = item[-1]
      if lang == 'usa':
        lang = 'us'
      spk2lang[spk_id] = lang
      spk2gender[spk_id] = gender
      spk2dataset[spk_id] = dataset

  for dataset in ['dev', 'test']:
    utt2wav = []
    utt2spk = []
    utt2lang = []
    spk2gender_ = []
    for spk_id in list(spk2lang.keys())[1:]:
      if spk2dataset[spk_id] != dataset:
        continue
      spk2gender_.append(spk_id + ' ' + spk2gender[spk_id] + '\n')
      video_ids = os.listdir(f'/data/pytong/wav/voxceleb1/{dataset}_wav/{spk_id}')
      for video_id in video_ids:
        audios = os.listdir(
          f'/data/pytong/wav/voxceleb1/{dataset}_wav/{spk_id}/{video_id}'
        )
        utts = [f"{spk_id}-{video_id.replace('-', '+')}{audio.split('.')[0]}" for audio in audios]
        paths = [f'/data/pytong/wav/voxceleb1/{dataset}_wav/{spk_id}/{video_id}/{audio}'
                for audio in audios]
        utt2wav.extend([utt + ' ' + path + '\n'
                        for (utt, path) in zip(utts, paths)])
        utt2spk.extend([utt + ' ' + spk_id + '\n' for utt in utts])
        utt2lang.extend([utt + ' ' + spk2lang[spk_id] + '\n' for utt in utts])
    with open(f'data/{dataset}/spk2gender', 'w') as f:
      f.writelines(line for line in spk2gender_)
    with open(f'data/{dataset}/wav.scp', 'w') as f:
      f.writelines(line for line in utt2wav)
    with open(f'data/{dataset}/utt2lang', 'w') as f:
      f.writelines(line for line in utt2lang)
    with open(f'data/{dataset}/utt2spk', 'w') as f:
      f.writelines(line for line in utt2spk)

def stat():
  utt_by_lang = dict()
  spk_by_lang = dict()
  male_by_lang = dict()
  female_by_lang = dict()
  lang2spk = dict()
  spk2gender = dict()
  for dataset in ['dev', 'test']:
    with open(f'data/{dataset}/utt2lang', 'r') as f:
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

    with open(f'data/{dataset}/spk2gender', 'r') as f:
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
  for dataset in ['dev', 'test']:
    with open(f'{dataset}/utt2lang', 'r') as f:
      for line in f:
        item = line.strip().split()
        utt = item[0]
        spk = utt.split('-')[0]
        lang = item[1]
        if lang in ['china', 'india', 'pakistan', 'singapore', 'france', 'iran',
                    'trinidad', 'uk', 'us']:
          utt2lang.append(utt + ' ' + lang + '\n')
          continue
        if lang == 'sweden' and spk != 'id10723':
          utt2lang.append(utt + ' ' + lang + '\n')
          continue
        if spk in ['id10392', 'id10903']:
          utt2lang.append(utt + ' ' + lang + '\n')
  with open('utt2lang.filtered', 'w') as f:
    f.writelines(line for line in utt2lang)


if __name__ == '__main__':
  utt_by_lang, spk_by_lang, male_by_lang, female_by_lang = stat()