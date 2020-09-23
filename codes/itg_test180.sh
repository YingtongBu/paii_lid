
utils/utt2spk_to_spk2utt.pl data/itg_test180/utt2spk > data/itg_test180/spk2utt
utils/fix_data_dir.sh data/itg_test180

stage=0

. ./cmd.sh
#
#--- VLTN ---
if [ $stage -eq 0 ]; then
#. ./cmd.sh;
    for t in itg_test180; do
        cp -r data/${t} data/${t}_novtln
        rm -r data/${t}_novtln/{split,.backup,spk2warp} 2>/dev/null || true
        steps/make_mfcc.sh --mfcc-config conf/mfcc_vtln.conf --nj 40 --cmd "$train_cmd" \
           data/${t}_novtln exp_37/make_mfcc mfcc
        lid/compute_vad_decision.sh data/${t}_novtln exp_37/make_mfcc mfcc
     done

    utils/fix_data_dir.sh data/itg_test180_novtln

#    . ./cmd.sh;
    for t in itg_test180; do
        lid/get_vtln_warps.sh --nj 8 --cmd "$train_cmd" \
           data/${t}_novtln exp_37/vtln exp_37/${t}_warps
        cp exp_37/${t}_warps/utt2warp data/$t/
    done
fi

#--- MFCC ---

if [ $stage -eq 1 ]; then
    utils/fix_data_dir.sh data/itg_test180
    utils/filter_scp.pl data/itg_test180/utt2warp data/itg_test180/utt2spk > data/itg_test180/utt2spk_tmp
    cp data/itg_test180/utt2spk_tmp data/itg_test180/utt2spk
    utils/fix_data_dir.sh data/itg_test180

    #. ./cmd.sh;
    mfccdir=`pwd`/mfcc;
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
      data/itg_test180 exp_37/make_mfcc $mfccdir


    #. ./cmd.sh;
    vaddir=`pwd`/mfcc;
    lid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/itg_test180 \
      exp_37/make_vad $vaddir


    #. ./cmd.sh;
    lid/extract_ivectors.sh --cmd "$train_cmd --mem 3G" --nj 8 \
       exp_37/extractor_2048_100 data/itg_test180 exp_37/ivectors_itg_test180_100

    #. ./cmd.sh;
    lid/extract_ivectors.sh --cmd "$train_cmd --mem 3G" --nj 8 \
       exp_37/extractor_2048_200 data/itg_test180 exp_37/ivectors_itg_test180_200

    #. ./cmd.sh;
    lid/extract_ivectors.sh --cmd "$train_cmd --mem 3G" --nj 8 \
       exp_37/extractor_2048_600 data/itg_test180 exp_37/ivectors_itg_test180_600
fi

if [ $stage -eq 2 ]; then
    mkdir ivectors.itg_test180_100
    for ((i=1;i<=8;i++)); do
      ../../../src/bin/copy-vector ark:exp_37/ivectors_itg_test180_100/ivector.$i.ark  ark,t:ivectors.itg_test180_100/ivector.$i.txt
    done
    mkdir ivectors.itg_test180_200
    for ((i=1;i<=8;i++)); do
      ../../../src/bin/copy-vector ark:exp_37/ivectors_itg_test180_200/ivector.$i.ark  ark,t:ivectors.itg_test180_200/ivector.$i.txt
    done
    mkdir ivectors.itg_test180_600
    for ((i=1;i<=8;i++)); do
      ../../../src/bin/copy-vector ark:exp_37/ivectors_itg_test180_600/ivector.$i.ark  ark,t:ivectors.itg_test180_600/ivector.$i.txt
    done
fi

if [ $stage -eq 2 ]; then
    lid/run_logistic_regression_edit.sh data_mturk_train5479 itg_test180 exp_37

fi
