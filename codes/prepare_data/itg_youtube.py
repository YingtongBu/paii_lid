
import os

utt2lang = []
utt2spk = []
utt2wav = []

re2vi = dict()
re2src = dict()

dataset = 'lre07'

with open(f'data/{dataset}_tmp/wav.scp', 'r') as f:
  for line in f:
    item = line.split()
    recording_id = item[0]
    source = item[1].split('/')[-2]
    video_id = item[1].split('/')[-1].split('.')[0]
    re2vi[recording_id] = video_id
    re2src[recording_id] = source

with open(f'data/{dataset}_tmp/utt2lang', 'r') as f:
  for line in f:
    item = line.split()
    utt = item[0]
    spk = utt.split('-')[0]
    recording_id = utt.split('-')[1]
    start = int(utt.split('-')[2][:4])
    seg_idx = f'{int(start/30)-1:03d}' # audio is cut after 30 seconds of original audios, so videoid_000.wav actually starts at 30s
    lang = item[1]
    source = re2src[recording_id]
    video_id = re2vi[recording_id]
    dir = f'/data/pytong/wav/youtube/{source}/{video_id}/{video_id}_{seg_idx}.wav'

    utt2lang.append(utt + ' ' + lang + '\n')
    utt2spk.append(utt + ' ' + spk + '\n')
    utt2wav.append(utt + ' ' + dir + '\n')


with open(f'data/{dataset}_youtube/wav.scp', 'w') as f:
  f.writelines(line for line in utt2wav)

with open(f'data/{dataset}_youtube/utt2lang', 'w') as f:
  f.writelines(line for line in utt2lang)

with open(f'data/{dataset}_youtube/utt2spk', 'w') as f:
  f.writelines(line for line in utt2spk)

os.system(f'cat data/{dataset}_youtube/utt2lang data/{dataset}_itg/utt2lang > data/{dataset}_all2/utt2lang')
os.system(f'cat data/{dataset}_youtube/utt2spk data/{dataset}_itg/utt2spk > data/{dataset}_all2/utt2spk')
os.system(f'cat data/{dataset}_youtube/wav.scp data/{dataset}_itg/wav.scp > data/{dataset}_all2/wav.scp')


# unique_spks = []
# with open('data/train_youtube/utt2lang') as f:
#   for line in f:
#     item = line.strip().split()
#     spk = item[0].split('-')[0]
#     if item[-1] == 'us':
#       continue
#     if spk in unique_spks:
#       continue
#     unique_spks.append(spk)
#
# with open('data/lre07_youtube/utt2lang') as f:
#   for line in f:
#     item = line.strip().split()
#     spk = item[0].split('-')[0]
#     if item[-1] == 'us':
#       continue
#     if spk in unique_spks:
#       continue
#     unique_spks.append(spk)
#
# print(len(unique_spks))