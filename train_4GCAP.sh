# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
lnss=(0)
paths=(5 15 50)
# FILTERED
for i in {0..9}
do
      for lns in "${lnss[@]}"
      do
            for mu in 1 3 9
            do
                  for opt in adam
                  do
                        for p in "${paths[@]}"
                        do
                        python train_imagenet_4.py \
                              --config-file rn50_configs/rn50_88_epochs.yaml \
                              --logging.folder=results/ \
                              --loss.loss=sigmoid \
                              --logging.log_level=1 \
                              --logging.experiment_name=12MOREGOOD \
                              --data.val_dataset=/mnt/lustre/work/oh/owl156/data/ImageNet4_8more_val_400_0.5_90.beton \
                              --loss.mu=$mu \
                              --loss.lmbd=0 \
                              --lr.lr=0.01 \
                              --lr.lr_schedule_type=cyclic \
                              --training.optimizer=$opt \
                              --training.use_blurpool=0 \
                              --data.train_dataset=/mnt/lustre/work/oh/owl156/data/12FILTEREDHARD_400_0.5_90_$p.beton \
                              --data.num_classes=12 \
                              --training.distributed=0 \
                              --training.negative_label_smoothing=$lns
                        done
                  done
            done
      done
done

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
lnss=(0)
paths=(15 50 150)
# FILTERED
for i in {0..9}
do
      for lns in "${lnss[@]}"
      do
            for opt in adam
            do
                  for p in "${paths[@]}"
                  do
                  python train_imagenet_4.py \
                        --config-file rn50_configs/rn50_88_epochs.yaml \
                        --logging.folder=results/ \
                        --loss.loss=sigmoid \
                        --logging.log_level=1 \
                        --logging.experiment_name=12MORE_RANDOM \
                        --data.val_dataset=/mnt/lustre/work/oh/owl156/data/ImageNet4_8more_val_400_0.5_90.beton \
                        --loss.mu=1 \
                        --loss.lmbd=0 \
                        --lr.lr=0.01 \
                        --lr.lr_schedule_type=cyclic \
                        --training.optimizer=$opt \
                        --training.use_blurpool=0 \
                        --data.train_dataset=/mnt/lustre/work/oh/owl156/data/12more_random_400_0.5_90_$p.beton \
                        --data.num_classes=12 \
                        --training.distributed=0 \
                        --training.negative_label_smoothing=$lns
                  done
            done
      done
done