#!/bin/bash

python train_imagenet_4.py \
      --config-file rn50_configs/rn50_88_epochs.yaml \
      --logging.folder=results/ \
      --loss.loss=sigmoid \
      --logging.log_level=1 \
      --logging.experiment_name=$1 \
      --data.val_dataset=$5 \
      --loss.mu=$7 \
      --loss.lmbd=$8 \
      --lr.lr=0.01 \
      --lr.lr_schedule_type=cyclic \
      --training.optimizer=$2 \
      --training.use_blurpool=0 \
      --data.train_dataset=$3 \
      --data.num_classes=$6 \
      --training.distributed=0 \
      --training.negative_label_smoothing=$4