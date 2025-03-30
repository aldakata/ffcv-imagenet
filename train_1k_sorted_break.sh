export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes=1 --standalone --nproc_per_node=8 --max_restarts=1 train_imagenet_elastic.py \
   --config-file rn50_configs/rn50_88_epochs.yaml \
   --logging.folder=results/ \
   --loss.loss=sigmoid \
   --logging.log_level=1 \
   --logging.experiment_name=BREAK_NEG_LOSS_NO_UPDATE \
   --data.val_dataset=/mnt/lustre/work/oh/owl156/HardNegativeSamples/ffcv-imagenet/val_500_0.50_90_2.ffcv \
   --loss.mu=1 \
   --loss.lmbd=0 \
   --lr.lr=1.5 \
   --lr.lr_schedule_type=cyclic \
   --training.optimizer=sgd \
   --training.use_blurpool=1 \
   --data.train_dataset=/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.sorted_150.beton \
   --training.negative_update=0

torchrun --nnodes=1 --standalone --nproc_per_node=8 --max_restarts=1 train_imagenet_elastic.py \
   --config-file rn50_configs/rn50_88_epochs.yaml \
   --logging.folder=results/ \
   --loss.loss=sigmoid \
   --logging.log_level=1 \
   --logging.experiment_name=BREAK_NEG_LOSS_NO_UPDATE \
   --data.val_dataset=/mnt/lustre/work/oh/owl156/HardNegativeSamples/ffcv-imagenet/val_500_0.50_90_2.ffcv \
   --loss.mu=1 \
   --loss.lmbd=0 \
   --lr.lr=1.5 \
   --lr.lr_schedule_type=cyclic \
   --training.optimizer=sgd \
   --training.use_blurpool=1 \
   --data.train_dataset=/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.sorted_150.beton \
   --training.negative_update=0


torchrun --nnodes=1 --standalone --nproc_per_node=8 --max_restarts=1 train_imagenet_elastic.py \
   --config-file rn50_configs/rn50_88_epochs.yaml \
   --logging.folder=results/ \
   --loss.loss=sigmoid \
   --logging.log_level=1 \
   --logging.experiment_name=BREAK_NEG_LOSS_NO_UPDATE \
   --data.val_dataset=/mnt/lustre/work/oh/owl156/HardNegativeSamples/ffcv-imagenet/val_500_0.50_90_2.ffcv \
   --loss.mu=10 \
   --loss.lmbd=0 \
   --lr.lr=1.5 \
   --lr.lr_schedule_type=cyclic \
   --training.optimizer=sgd \
   --training.use_blurpool=1 \
   --data.train_dataset=/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.sorted_150.beton \
   --training.negative_update=0

torchrun --nnodes=1 --standalone --nproc_per_node=8 --max_restarts=1 train_imagenet_elastic.py \
   --config-file rn50_configs/rn50_88_epochs.yaml \
   --logging.folder=results/ \
   --loss.loss=sigmoid \
   --logging.log_level=1 \
   --logging.experiment_name=BREAK_NEG_LOSS_NO_UPDATE \
   --data.val_dataset=/mnt/lustre/work/oh/owl156/HardNegativeSamples/ffcv-imagenet/val_500_0.50_90_2.ffcv \
   --loss.mu=10 \
   --loss.lmbd=0 \
   --lr.lr=1.5 \
   --lr.lr_schedule_type=cyclic \
   --training.optimizer=sgd \
   --training.use_blurpool=1 \
   --data.train_dataset=/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.sorted_150.beton \
   --training.negative_update=0

torchrun --nnodes=1 --standalone --nproc_per_node=8 --max_restarts=1 train_imagenet_elastic.py \
   --config-file rn50_configs/rn50_88_epochs.yaml \
   --logging.folder=results/ \
   --loss.loss=sigmoid \
   --logging.log_level=1 \
   --logging.experiment_name=BREAK_NEG_LOSS_NO_UPDATE \
   --data.val_dataset=/mnt/lustre/work/oh/owl156/HardNegativeSamples/ffcv-imagenet/val_500_0.50_90_2.ffcv \
   --loss.mu=100 \
   --loss.lmbd=0 \
   --lr.lr=1.5 \
   --lr.lr_schedule_type=cyclic \
   --training.optimizer=sgd \
   --training.use_blurpool=1 \
   --data.train_dataset=/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.sorted_150.beton \
   --training.negative_update=0

torchrun --nnodes=1 --standalone --nproc_per_node=8 --max_restarts=1 train_imagenet_elastic.py \
   --config-file rn50_configs/rn50_88_epochs.yaml \
   --logging.folder=results/ \
   --loss.loss=sigmoid \
   --logging.log_level=1 \
   --logging.experiment_name=BREAK_NEG_LOSS_NO_UPDATE \
   --data.val_dataset=/mnt/lustre/work/oh/owl156/HardNegativeSamples/ffcv-imagenet/val_500_0.50_90_2.ffcv \
   --loss.mu=100 \
   --loss.lmbd=0 \
   --lr.lr=1.5 \
   --lr.lr_schedule_type=cyclic \
   --training.optimizer=sgd \
   --training.use_blurpool=1 \
   --data.train_dataset=/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.sorted_150.beton \
   --training.negative_update=0

torchrun --nnodes=1 --standalone --nproc_per_node=8 --max_restarts=1 train_imagenet_elastic.py \
   --config-file rn50_configs/rn50_88_epochs.yaml \
   --logging.folder=results/ \
   --loss.loss=sigmoid \
   --logging.log_level=1 \
   --logging.experiment_name=BREAK_NEG_LOSS_NO_UPDATE \
   --data.val_dataset=/mnt/lustre/work/oh/owl156/HardNegativeSamples/ffcv-imagenet/val_500_0.50_90_2.ffcv \
   --loss.mu=1000 \
   --loss.lmbd=0 \
   --lr.lr=1.5 \
   --lr.lr_schedule_type=cyclic \
   --training.optimizer=sgd \
   --training.use_blurpool=1 \
   --data.train_dataset=/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.sorted_150.beton \
   --training.negative_update=0

torchrun --nnodes=1 --standalone --nproc_per_node=8 --max_restarts=1 train_imagenet_elastic.py \
   --config-file rn50_configs/rn50_88_epochs.yaml \
   --logging.folder=results/ \
   --loss.loss=sigmoid \
   --logging.log_level=1 \
   --logging.experiment_name=BREAK_NEG_LOSS_NO_UPDATE \
   --data.val_dataset=/mnt/lustre/work/oh/owl156/HardNegativeSamples/ffcv-imagenet/val_500_0.50_90_2.ffcv \
   --loss.mu=1000 \
   --loss.lmbd=0 \
   --lr.lr=1.5 \
   --lr.lr_schedule_type=cyclic \
   --training.optimizer=sgd \
   --training.use_blurpool=1 \
   --data.train_dataset=/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.sorted_150.beton \
   --training.negative_update=0
