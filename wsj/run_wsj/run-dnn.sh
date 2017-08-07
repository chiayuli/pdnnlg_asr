#!/bin/bash

# Copyright 2017   Chia-Yu Li  University of Stuttgart      Apache 2.0
# This is the script  that trains DNN model over fMLLR features.  It is to be
# run after run.sh. Before running this, you should already build the initial
# GMM model. This script requires a GPU, and also the "pdnnlg" toolkit to train
# the DNN.

working_dir=exp_pdnn/dnn
gmmdir=exp/tri4b

# Specify the gpu device to be used
gpu=gpu

. cmd.sh
[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if ! nvidia-smi; then
  echo "The command nvidia-smi was not found: this probably means you don't have a GPU."
  echo "(Note: this script might still work, it would just be slower.)"
fi

# The hope here is that Theano has been installed either to python or to python2.6
pythonCMD=python
if ! python -c 'import theano;'; then
  if ! python2.6 -c 'import theano;'; then
    echo "Theano does not seem to be installed on your machine.  Not continuing."
    echo "(Note: this script might still work, it would just be slower.)"
    exit 1;
  else
    pythonCMD=python2.6
  fi
fi

mkdir -p $working_dir/log

! gmm-info $gmmdir/final.mdl >&/dev/null && \
   echo "Error getting GMM info from $gmmdir/final.mdl" && exit 1;

num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'` || exit 1;

echo =====================================================================
echo "           Data Split & Alignment & Feature Preparation            "
echo =====================================================================
# Split training data into traing and cross-validation sets for DNN
if [ ! -d data/train_tr95 ]; then
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train_si284 data/train_tr95 data/train_cv05 || exit 1
fi
# Alignment on the training and validation data. We set --nj to 14 because data/train_cv05 has 14 speakers.
for set in tr95 cv05; do
  if [ ! -d ${gmmdir}_ali_$set ]; then
    steps/align_fmllr.sh --nj 14   \
      data/train_$set data/lang $gmmdir ${gmmdir}_ali_$set || exit 1
  fi
done
# Dump fMLLR features. "Fake" cmvn states (0 means and 1 variance) are applied. 
for set in tr95 cv05; do
  if [ ! -d $working_dir/data/train_$set ]; then
    steps/nnet/make_fmllr_feats.sh --nj 14   \
      --transform-dir ${gmmdir}_ali_$set \
      $working_dir/data/train_$set data/train_$set $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data/train_$set $working_dir/_log $working_dir/_fmllr || exit 1;
  fi
done
for set in dev93 eval92; do
  if [ ! -d $working_dir/data/$set ]; then
    steps/nnet/make_fmllr_feats.sh --nj 8   \
      --transform-dir $gmmdir/decode_bd_tgpr_$set \
      $working_dir/data/$set data/test_$set $gmmdir $working_dir/_log $working_dir/_fmllr || exit 1
    steps/compute_cmvn_stats.sh --fake \
      $working_dir/data/$set $working_dir/_log $working_dir/_fmllr || exit 1;
  fi
done

echo =====================================================================
echo "               Training and Cross-Validation Pfiles                "
echo =====================================================================
# By default, DNN inputs include 11 frames of fMLLR
for set in tr95 cv05; do
  if [ ! -f $working_dir/${set}.pfile.done ]; then
    steps_pdnnlg/build_nnet_pfile.sh   --norm-vars false --do-concat false \
      --splice-opts "--left-context=5 --right-context=5" \
      $working_dir/data/train_$set ${gmmdir}_ali_$set $working_dir || exit 1
    touch $working_dir/${set}.pfile.done
  fi
done

echo =====================================================================
echo "                          DNN Training                             "
echo =====================================================================
feat_dim=$(gunzip -c $working_dir/train_tr95.pfile.1.gz |head |grep num_features| awk '{print $2}') || exit 1;

if [ ! -f $working_dir/dnn.fine.done ]; then
  echo "Fine-tuning DNN"
  export CUDA_VISIBLE_DEVICES="0";
  export PYTHONPATH=$PYTHONPATH:`pwd`/pdnnlg/ \; \
  export THEANO_FLAGS=mode=FAST_RUN,device=$gpu,floatX=float32 \; \
  $pythonCMD pdnnlg/cmds/run_DNN.py --train-data "$working_dir/train_tr95.pfile.*.gz,partition=2000m,random=true,stream=true" \
                                    --valid-data "$working_dir/train_cv05.pfile.*.gz,partition=600m,random=true,stream=true" \
                                    --nnet-spec "$feat_dim:1024:1024:1024:1024:1024:1024:$num_pdfs" \
                                    --lrate "D:0.08:0.5:0.2,0.2:8" \
                                    --wdir $working_dir --kaldi-output-file $working_dir/dnn.nnet || exit 1;
  touch $working_dir/dnn.fine.done
fi

echo =====================================================================
echo "                           Decoding                                "
echo =====================================================================
if [ ! -f  $working_dir/decode.done ]; then
  cp $gmmdir/final.mdl $working_dir || exit 1;  # copy final.mdl for scoring
  graph_dir=$gmmdir/graph_nosp_bd_tgpr
  steps_pdnnlg/decode_dnn.sh --nj 10 --scoring-opts "--min-lmwt 7 --max-lmwt 18" \
    $graph_dir $working_dir/data/dev93 ${gmmdir}_ali_tr95 $working_dir/decode_bd_tgpr_dev93 $working_dir || exit 1;
  steps_pdnnlg/decode_dnn.sh --nj 8 --scoring-opts "--min-lmwt 7 --max-lmwt 18"  \
    $graph_dir $working_dir/data/eval92 ${gmmdir}_ali_tr95 $working_dir/decode_bd_tgpr_eval92 $working_dir || exit 1;
  touch $working_dir/decode.done
fi

echo "Finish !!"
