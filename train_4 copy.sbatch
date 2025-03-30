#!/bin/bash

#SBATCH --job-name=5minjob         # Job name
#SBATCH --ntasks=1                 # Number of tasks (see below)
#SBATCH --nodes=1                  # Ensure that all cores are on one machine
#SBATCH --time=00:59:00             # Runtime in D-HH:MM
#SBATCH --gres=gpu:1               # optionally type and number of gpus
#SBATCH --output=job_logs_4/%x.%j.out   # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=job_logs_4/%x.%j.err    # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --partition=a100-galvani     # Partition to submit to

####
#c) Execute your file.
####
source /mnt/lustre/work/oh/owl156/miniforge3/etc/profile.d/conda.sh
conda activate ffcv_cuda
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