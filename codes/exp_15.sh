#!/usr/bin/env bash
#-------------------------------------------------
#youtube_v7 16khz 9:1 num-ceps=5, num-mel-bins=20
#-------------------------------------------------

stage=0

# 6 class categories
vim local/general_lr_closed_set_langs.txt

awk '{print $2}' data/train_youtube_v7/utt2lang | sort | uniq -c | sort -nr
#   2796 uk
#   2393 us

utils/utt2spk_to_spk2utt.pl data/train_youtube_v7/utt2spk > data/train_youtube_v7/spk2utt
utils/utt2spk_to_spk2utt.pl data/lre07_youtube_v7/utt2spk > data/lre07_youtube_v7/spk2utt

utils/fix_data_dir.sh data/train_youtube_v7
utils/fix_data_dir.sh data/lre07_youtube_v7
5189 / 510

--- VLTN ---

vim conf/mfcc.conf
vim conf/mfcc_vtln.conf

--sample-frequency=16000
--frame-length=25 # the default is 25.
--low-freq=20 # the default.
--high-freq=7600 # the default is zero meaning use the Nyquist (4k in this case).
--num-ceps=5
--num-mel-bins=20
--allow-downsample=true
--allow-upsample=true


. ./cmd.sh;
for t in train_youtube_v7 lre07_youtube_v7; do
    cp -r data/${t} data/${t}_novtln
    rm -r data/${t}_novtln/{split,.backup,spk2warp} 2>/dev/null || true
    steps/make_mfcc.sh --mfcc-config conf/mfcc_vtln.conf --nj 80 --cmd "$train_cmd" \
       data/${t}_novtln exp_15/make_mfcc /home/puyingtong/kaldi/egs/lre07/v1/mfcc
    lid/compute_vad_decision.sh data/${t}_novtln exp_15/make_mfcc /home/puyingtong/kaldi/egs/lre07/v1/mfcc
 done


utils/fix_data_dir.sh data/train_youtube_v7_novtln
utils/fix_data_dir.sh data/lre07_youtube_v7_novtln
5188 / 510

utils/subset_data_dir.sh data/train_youtube_v7_novtln 4000 data/train_youtube_v7_4k_novtln

. ./cmd.sh;
sid/train_diag_ubm.sh --nj 30 --cmd "$train_cmd" data/train_youtube_v7_4k_novtln 256 \
    exp_15/diag_ubm_vtln
lid/train_lvtln_model.sh --mfcc-config conf/mfcc_vtln.conf --nj 30 --cmd "$train_cmd" \
     data/train_youtube_v7_4k_novtln exp_15/diag_ubm_vtln exp_15/vtln
# warning

. ./cmd.sh;
for t in lre07_youtube_v7 train_youtube_v7; do
    lid/get_vtln_warps.sh --nj 10 --cmd "$train_cmd" \
       data/${t}_novtln exp_15/vtln exp_15/${t}_warps
    cp exp_15/${t}_warps/utt2warp data/$t/
done


--- MFCC ---

utils/fix_data_dir.sh data/train_youtube_v7
utils/filter_scp.pl data/train_youtube_v7/utt2warp data/train_youtube_v7/utt2spk > data/train_youtube_v7/utt2spk_tmp
cp data/train_youtube_v7/utt2spk_tmp data/train_youtube_v7/utt2spk
utils/fix_data_dir.sh data/train_youtube_v7
5181

utils/fix_data_dir.sh data/lre07_youtube_v7
utils/filter_scp.pl data/lre07_youtube_v7/utt2warp data/lre07_youtube_v7/utt2spk > data/lre07_youtube_v7/utt2spk_tmp
cp data/lre07_youtube_v7/utt2spk_tmp data/lre07_youtube_v7/utt2spk
utils/fix_data_dir.sh data/lre07_youtube_v7


. ./cmd.sh;
mfccdir=`pwd`/mfcc;
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/train_youtube_v7 exp_15/make_mfcc $mfccdir
. ./cmd.sh;
mfccdir=`pwd`/mfcc;
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/lre07_youtube_v7 exp_15/make_mfcc $mfccdir



. ./cmd.sh;
vaddir=`pwd`/mfcc;
lid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/train_youtube_v7 \
  exp_15/make_vad $vaddir
. ./cmd.sh;
vaddir=`pwd`/mfcc;
lid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/lre07_youtube_v7 \
  exp_15/make_vad $vaddir

--- ubm ---

utils/subset_data_dir.sh data/train_youtube_v7 1000 data/train_youtube_v7_1k
utils/subset_data_dir.sh data/train_youtube_v7 4000 data/train_youtube_v7_4k


. ./cmd.sh;
lid/train_diag_ubm.sh --nj 30 --cmd "$train_cmd --mem 20G" \
  data/train_youtube_v7_1k 2048 exp_15/diag_ubm_2048
. ./cmd.sh;
lid/train_full_ubm.sh --nj 30 --cmd "$train_cmd --mem 20G" \
  data/train_youtube_v7_4k exp_15/diag_ubm_2048 exp_15/full_ubm_2048_4k

. ./cmd.sh;
lid/train_full_ubm.sh --nj 30 --cmd "$train_cmd --mem 35G" \
  data/train_youtube_v7 exp_15/full_ubm_2048_4k exp_15/full_ubm_2048


--- i-vector ---

nohup lid/train_ivector_extractor.sh --cmd "run.pl --mem 4G --mem 35G" \
  --stage 1 \
  --use-weights true \
  --num-iters 5 exp_15/full_ubm_2048/final.ubm data/train_youtube_v7 \
  exp_15/extractor_2048 >> train_ivector.log15 &



cp -r data/train_youtube_v7 data/train_youtube_v7_lr
languages=local/general_lr_closed_set_langs.txt
utils/filter_scp.pl -f 2 $languages <(lid/remove_dialect.pl data/train_youtube_v7/utt2lang) \
  > data/train_youtube_v7_lr/utt2lang
utils/fix_data_dir.sh data/train_youtube_v7_lr
5181

awk '{print $2}' data/train_youtube_v7_lr/utt2lang | sort | uniq -c | sort -nr
   2790 uk
   2391 us


nohup lid/extract_ivectors.sh --cmd "run.pl --mem 4G --mem 3G" --nj 30 \
   exp_15/extractor_2048 data/train_youtube_v7_lr exp_15/ivectors_train_youtube_v7 > extract_ivector.log3 &

nohup lid/extract_ivectors.sh --cmd "run.pl --mem 4G --mem 3G" --nj 10 \
   exp_15/extractor_2048 data/lre07_youtube_v7 exp_15/ivectors_lre07_youtube_v7 > extract_ivector.log4 &


lid/run_logistic_regression_edit.sh train_youtube_v7 lre07_youtube_v7 exp_15

~/kaldi/src/bin/compute-wer --text ark:<(lid/remove_dialect.pl data/lre07_youtube_v7/utt2lang) \
  ark:exp_15/ivectors_lre07_youtube_v7/output

~/kaldi/src/bin/compute-wer --mode=present --text ark:<(lid/remove_dialect.pl data/train_youtube_v7_lr/utt2lang) \
  ark:exp_15/ivectors_train_youtube_v7/output

