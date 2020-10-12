#!/usr/bin/env bash

# This script is for ivector training and extract ivector feature for other model training
#-------------------------------------------------
#train_all / test_1368
#-------------------------------------------------
stage=0
train_all=train_0903
test_1368=test_1368
#*** Fix data ***
. ./cmd.sh;
awk '{print $2}' data/$train_all/utt2lang | sort | uniq -c | sort -nr
   # 1742 us
   # 1082 nnl
   #  986 uk
   #  831 nnm
   #  548 nnn
   #  290 nnh
if [ $stage -eq 0 ]; then
    utils/utt2spk_to_spk2utt.pl data/$train_all/utt2spk > data/$train_all/spk2utt
    utils/utt2spk_to_spk2utt.pl data/$test_1368/utt2spk > data/$test_1368/spk2utt

    utils/fix_data_dir.sh data/$train_all
    utils/fix_data_dir.sh data/$test_1368
fi
#--- VTLN ---
if [ $stage -eq 1 ]; then
#    . ./cmd.sh;
    for t in $train_all $test_1368; do
        cp -r data/${t} data/${t}_novtln
        rm -r data/${t}_novtln/{split,.backup,spk2warp} 2>/dev/null || true
        steps/make_mfcc.sh --mfcc-config conf/mfcc_vtln.conf --nj 12 --cmd "$train_cmd" \
           data/${t}_novtln exp_38/make_mfcc mfcc
        lid/compute_vad_decision.sh data/${t}_novtln exp_38/make_mfcc mfcc
     done

    utils/fix_data_dir.sh data/${train_all}_novtln
    utils/fix_data_dir.sh data/${test_1368}_novtln
    #5477
    #1368

    utils/subset_data_dir.sh data/${train_all}_novtln 2000 data/${train_all}_2k_novtln

#    . ./cmd.sh;
    sid/train_diag_ubm.sh --nj 30 --cmd "$train_cmd" data/${train_all}_2k_novtln 256 \
        exp_38/diag_ubm_vtln
    lid/train_lvtln_model.sh --mfcc-config conf/mfcc_vtln.conf --nj 30 --cmd "$train_cmd" \
         data/${train_all}_2k_novtln exp_38/diag_ubm_vtln exp_38/vtln

#    . ./cmd.sh;
    for t in $test_1368 $train_all; do
        lid/get_vtln_warps.sh --nj 50 --cmd "$train_cmd" \
           data/${t}_novtln exp_38/vtln exp_38/${t}_warps
        cp exp_38/${t}_warps/utt2warp data/$t/
    done

fi

#--- MFCC ---
if [ $stage -eq 2 ]; then
    utils/fix_data_dir.sh data/$train_all
    utils/filter_scp.pl data/$train_all/utt2warp data/$train_all/utt2spk > data/$train_all/utt2spk_tmp
    cp data/$train_all/utt2spk_tmp data/$train_all/utt2spk
    utils/fix_data_dir.sh data/$train_all
    5470

    #. ./cmd.sh;
    mfccdir=`pwd`/mfcc;
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 80 --cmd "$train_cmd" \
      data/$train_all exp_38/make_mfcc $mfccdir


    utils/fix_data_dir.sh data/$test_1368
    utils/filter_scp.pl data/$test_1368/utt2warp data/$test_1368/utt2spk > data/$test_1368/utt2spk_tmp
    cp data/$test_1368/utt2spk_tmp data/$test_1368/utt2spk
    utils/fix_data_dir.sh data/$test_1368


#    . ./cmd.sh;
    mfccdir=`pwd`/mfcc;
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/$test_1368 exp_38/make_mfcc $mfccdir


#    . ./cmd.sh;

    vaddir=`pwd`/mfcc;
    lid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/$train_all \
      exp_38/make_vad $vaddir
#    . ./cmd.sh;

    vaddir=`pwd`/mfcc;
    lid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/$test_1368 \
      exp_38/make_vad $vaddir

    utils/subset_data_dir.sh data/$train_all 1000 data/${train_all}_1k
    utils/subset_data_dir.sh data/$train_all 2000 data/${train_all}_2k
fi

#. ./cmd.sh;
if [ $stage -eq 3 ]; then
    lid/train_diag_ubm.sh --nj 30 --cmd "$train_cmd --mem 20G" \
      data/${train_all}_1k 2048 exp_38/diag_ubm_2048
fi
#. ./cmd.sh;
if [ $stage -eq 4 ]; then
    lid/train_full_ubm.sh --nj 30 --cmd "$train_cmd --mem 20G" \
      data/${train_all}_2k exp_38/diag_ubm_2048 exp_38/full_ubm_2048_2k
fi
#. ./cmd.sh;
if [ $stage -eq 5 ]; then
    lid/train_full_ubm.sh --nj 30 --cmd "$train_cmd --mem 35G" \
      data/${train_all} exp_38/full_ubm_2048_2k exp_38/full_ubm_2048
fi
if [ $stage -eq 6 ]; then
    nohup lid/train_ivector_extractor.sh --cmd "run.pl --mem 35G" \
      --use-weights true \
      --num-iters 5 exp_38/full_ubm_2048/final.ubm data/$train_all \
      exp_38/extractor_2048 >> ${train_all}_ivector.log38 &
fi

if [ $stage -eq 7 ]; then
    cp -r data/${train_all} data/${train_all}_lr
    languages=local/general_lr_closed_set_langs.txt
    utils/filter_scp.pl -f 2 $languages <(lid/remove_dialect.pl data/$train_all/utt2lang) \
      > data/${train_all}_lr/utt2lang
    utils/fix_data_dir.sh data/${train_all}_lr

fi
#5470
if [ $stage -eq 8 ]; then
awk '{print $2}' data/${train_all}_lr/utt2lang | sort | uniq -c | sort -nr
#   1742 us
#   1079 nnl
#    986 uk
#    829 nnm
#    545 nnn
#    289 nnh

awk '{print $2}' data/$test_1368/utt2lang | sort | uniq -c | sort -nr
#    436 us
#    267 nnl
#    247 uk
#    208 nnm
#    137 nnn
#     73 nnh

fi

if [ $stage -eq 9 ]; then
nohup lid/extract_ivectors.sh --cmd "run.pl --mem 3G" --nj 5 \
   exp_38/extractor_2048 data/${train_all}_lr exp_38/ivectors_${train_all} &> extract_ivector.log1 &

nohup lid/extract_ivectors.sh --cmd "run.pl --mem 3G" --nj 3 \
   exp_38/extractor_2048 data/$test_1368 exp_38/ivectors_${test_1368} &> extract_ivector.log2 &
fi

if [ $stage -eq 10 ]; then
lid/run_logistic_regression_edit.sh $train_all $test_1368 exp_38
fi
