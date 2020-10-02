#!/usr/bin/env bash

#dataset name:
train_set=data_train_itg_0903_xvector
test_set=data_train_itg_0903_xvector

stage=0
subset_sample=2000

. ./cmd.sh
. ./path.sh

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

nnet_dir=exp_5/xvector_nnet_1a

# vim conf/mfcc.conf
conf=16khz
musan_dir=/data/pytong/data/musan
#--- data preparation ---

# mfccdir=`pwd`/mfcc;

# if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then

#   utils/create_split_dir.pl \
#   /export/b{14,15,16,17}/$USER/kaldi-data/egs/sre16/v2/xvector-$(date +'%m_%d_%H_%M')/mfccs/storage $mfccdir/storage
# fi

#def uttaddlang():
#  spk2lang = {}
#  with open('utt2lang', 'r') as f:
#    for line in f:
#      item = line.strip().split()
#      utt = item[0]
#      spk = utt.split('-')[0]
#      lang = item[1]
#      spk2lang[spk] = lang
#  utt2dir = []
#  utt2spk = []
#  with open('wav.scp.bak', 'r') as f:
#    for line in f:
#      item = line.strip().split()
#      utt = item[0]
#      dir = item[1]
#      spk = utt.split('-')[0]
#      id = utt.split('-')[1]
#      lang = spk2lang[spk]
#      utt = f'{lang}-{spk}.{id}'
#      utt2dir.append(utt + ' ' + dir + '\n')
#      utt2spk.append(utt + ' ' + lang + '\n')
#  with open('wav.scp', 'w') as f:
#    f.writelines(line for line in utt2dir)
#  with open('utt2spk', 'w') as f:
#    f.writelines(line for line in utt2spk)

if [ $stage -eq 0 ]; then
#organize train_set &test_set; generate utt2spk and spk2utt 
  for name in $train_set $test_set; do
      utils/utt2spk_to_spk2utt.pl data/${name}/utt2spk > data/${name}/spk2utt
      utils/fix_data_dir.sh data/${name}
    done
#5479 / 1368
fi

. ./cmd.sh;

if [ $stage -eq 1 ]; then
  #extract mfcc feature for train_set & test_set
  for name in $train_set $test_set; do
      utils/fix_data_dir.sh data/${name}
      steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
        data/${name} exp_5/make_mfcc $mfccdir
      utils/fix_data_dir.sh data/${name}
      sid/compute_vad_decision.sh --nj 6 --cmd "$train_cmd" \
        data/${name} exp_5/make_vad $vaddir
      utils/fix_data_dir.sh data/${name}
    done
fi

if [ $stage -eq 2 ]; then

  #prepare for data augmentation
  frame_shift=0.01;
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/$train_set/utt2num_frames > data/$train_set/reco2dur

  #--- data aug ---
  # aug by reverbing (not adding any noises here)
  rvb_opts=();
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list");
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list");
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/$train_set data/${train_set}_reverb

  #Number of RIRs is 40000

  cp data/$train_set/vad.scp data/${train_set}_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/${train_set}_reverb data/${train_set}_reverb.new
  rm -rf data/${train_set}_reverb
  mv data/${train_set}_reverb.new data/${train_set}_reverb

  # Augment with musan
  steps/data/make_musan.sh --sampling-rate 16000 $musan_dir data
  for name in speech noise music; do
      utils/data/get_utt2dur.sh data/musan_${name}
      mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
    done

  echo "generate all types of data and combined together"
  #generate all types of data and combined together
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/$train_set data/${train_set}_noise

  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/$train_set data/${train_set}_music

  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/$train_set data/${train_set}_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/${train_set}_aug data/${train_set}_reverb data/${train_set}_noise data/${train_set}_music data/${train_set}_babble
fi 

if [ $stage -eq 3 ]; then

  # Take a random subset of the augmentations

  utils/subset_data_dir.sh data/${train_set}_aug $subset_sample data/${train_set}_aug_${subset_sample}
  utils/fix_data_dir.sh data/${train_set}_aug_${subset_sample}
  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/${train_set}_aug_${subset_sample} exp_5/make_mfcc $mfccdir
  #first to fix the subset dataset
  utils/fix_data_dir.sh data/${train_set}_aug_${subset_sample}
  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/${train_set}_combined data/${train_set}_aug_${subset_sample} data/$train_set
#12869
fi

# Now we prepare the features to generate examples for xvector training.
#--- xvector ---
if [ $stage -eq 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  echo "prepare feature for xvector training"
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 2 --cmd "$train_cmd" \
      data/${train_set}_combined data/${train_set}_combined_no_sil exp_5/${train_set}_combined_no_sil
  utils/fix_data_dir.sh data/${train_set}_combined_no_sil
fi
  # 14477
 
if [ $stage -eq 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  echo "restrict utt length"
   # restrict utt len
  mv data/${train_set}_combined_no_sil/utt2num_frames data/${train_set}_combined_no_sil/utt2num_frames.bak

  min_len=500
  awk -v min_len=${min_len} '{print $1, $2}' data/${train_set}_combined_no_sil/utt2num_frames.bak > data/${train_set}_combined_no_sil/utt2num_frames

  utils/filter_scp.pl data/${train_set}_combined_no_sil/utt2num_frames data/${train_set}_combined_no_sil/utt2spk > data/${train_set}_combined_no_sil/utt2spk.new
  mv data/${train_set}_combined_no_sil/utt2spk.new data/${train_set}_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/${train_set}_combined_no_sil

  # delete utterance whose spk utt number < 8
  awk '{print $1, NF-1}' data/${train_set}_combined_no_sil/spk2utt > data/${train_set}_combined_no_sil/spk2num
  # nnh 764
  # nnl 2853
  # nnm 2188
  # nnn 1441
  # uk 2608
  # us 4603

  min_num_utts=82;
  awk -v min_num_utts=${min_num_utts} '{print $1, $2}' data/${train_set}_combined_no_sil/spk2num | utils/filter_scp.pl - data/${train_set}_combined_no_sil/spk2utt > data/${train_set}_combined_no_sil/spk2utt.new

  mv data/${train_set}_combined_no_sil/spk2utt.new data/${train_set}_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/${train_set}_combined_no_sil/spk2utt > data/${train_set}_combined_no_sil/utt2spk

  utils/filter_scp.pl data/${train_set}_combined_no_sil/utt2spk data/${train_set}_combined_no_sil/utt2num_frames > data/${train_set}_combined_no_sil/utt2num_frames.new
  mv data/${train_set}_combined_no_sil/utt2num_frames.new data/${train_set}_combined_no_sil/utt2num_frames
  utils/fix_data_dir.sh data/${train_set}_combined_no_sil

fi 
if [ $stage -eq 6 ]; then
  # Stages 6 through 8 are handled in run_xvector.sh
  echo "start to train xvector model"
  # nnet_dir=exp_5/xvector_nnet_1a_200;
  # CUDA_VISIBLE_DEVICES=1,2 
  local/nnet3/xvector/run_xvector.sh --cmd "run.pl --gpu4" --stage 0 --train-stage -1 \
    --data data/${train_Set}_combined_no_sil --nnet-dir $nnet_dir \
    --egs-dir $nnet_dir/egs
fi 

if [ $stage -eq 9 ]; then
  # nnet_dir=exp_5/xvector_nnet_1a;
  # CUDA_VISIBLE_DEVICES=2,3 
   # Extract x-vectors for training dataset
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 2 \
      $nnet_dir data/${train_set}_combined \
      exp_5/xvectors_${train_set}_combined

  # nnet_dir=exp_5/xvector_nnet_1a_200;
  # CUDA_VISIBLE_DEVICES=0,1 
   # Extract x-vectors used in the evaluation.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 2 \
      $nnet_dir data/$test_set \
      exp_5/xvectors_${test_set}_200
fi 

#start to do logistic regression
# --- lr ---
if [ $stage -eq 20 ]; then
  lid/run_logistic_regression_xvector.sh ${train_set}_combined $test_set exp_5
  compute-wer --mode=present --text ark:<(lid/remove_dialect.pl data/test_mturk_1368/utt2spk)   ark:exp_5/xvectors_${test_set}/output

  mkdir xvectors.${train_set}_512
  mkdir xvectors.${train_set}_200
  ../../../src/bin/copy-vector ark:exp_5/xvectors_${train_set}_combined/xvector.1.ark  ark,t:xvectors.${train_set}_512/xvector.1.txt
  ../../../src/bin/copy-vector ark:exp_5/xvectors_${train_set}_combined/xvector.2.ark  ark,t:xvectors.${train_set}_512/xvector.2.txt

  mkdir xvectors.${test_set}_512
  mkdir xvectors.${test_set}_200
  ../../../src/bin/copy-vector ark:exp_5/xvectors_${test_set}_200/xvector.1.ark  ark,t:xvectors.${test_set}_200/xvector.1.txt
  ../../../src/bin/copy-vector ark:exp_5/xvectors_${test_set}_200/xvector.2.ark  ark,t:xvectors.${test_set}_200/xvector.2.txt

  mkdir xvectors.${train_set}_plp_512
  for ((i=1;i<=4;i++)); do
    ../../../src/bin/copy-vector ark:xvectors_train/xvector.$i.ark  ark,t:xvectors.${train_set}_plp_512/xvector.$i.txt
  done

  mkdir xvectors.${test_set}_plp_512
  for ((i=1;i<=4;i++)); do
    ../../../src/bin/copy-vector ark:xvectors_lre07/xvector.$i.ark  ark,t:xvectors.${test_set}_plp_512/xvector.$i.txt
  done
fi 