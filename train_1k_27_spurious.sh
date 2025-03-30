export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for seed in {0..2}
do
   for value in 15 50 75
   do
      torchrun --nnodes=1 --standalone --nproc_per_node=8 --max_restarts=1 train_imagenet_elastic.py \
         --config-file rn50_configs/rn50_88_epochs.yaml \
         --logging.folder=results/ \
         --loss.loss=sigmoid \
         --logging.log_level=1 \
         --logging.experiment_name=SPURIOUS \
         --data.val_dataset=/mnt/lustre/work/oh/owl156/HardNegativeSamples/ffcv-imagenet/val_500_0.50_90_2.ffcv \
         --loss.mu=30 \
         --loss.lmbd=0 \
         --lr.lr=0.001 \
         --lr.lr_schedule_type=cyclic \
         --training.optimizer=adam \
         --training.use_blurpool=1 \
         --data.train_dataset=/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.27neg_spu_${value}.beton \
         --training.negative_update=1
   done
done