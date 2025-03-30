#!/bin/bash

write_dataset () {
    echo "Writing ImageNet"
    python write_imagenet_negative_1k.py \
        --cfg.dataset=imagenet \
        --cfg.split=${1} \
        --cfg.positive_path=/mnt/lustre/datasets/ImageNet2012/${1} \
        --cfg.max_resolution=${2} \
        --cfg.write_mode=proportion \
        --cfg.compress_probability=${3} \
        --cfg.jpeg_quality=${4} \
        --cfg.neg_counts=$5 \
        --cfg.negative_path="/mnt/lustre/work/oh/owl156/data/27_1k_spurious" \
        --cfg.write_path=${6} \
        --cfg.subset_file_dst=${7} \
        --cfg.superset_file_src=${8} 
}
# write_dataset train 500 0.5 90 5  "/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.27neg_spu_5.beton"  "/mnt/lustre/work/oh/owl156/data/negative_paths_log/27neg_spu_5.log" ""
# write_dataset train 500 0.5 90 15  "/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.27neg_spu_15.beton"  "/mnt/lustre/work/oh/owl156/data/negative_paths_log/27neg_spu_15.log" ""
# write_dataset train 500 0.5 90 50  "/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.27neg_spu_50.beton"  "/mnt/lustre/work/oh/owl156/data/negative_paths_log/27neg_spu_50.log" "/mnt/lustre/work/oh/owl156/data/negative_paths_log/27neg_spu_15.log"
write_dataset train 500 0.5 90 75 "/mnt/lustre/work/oh/owl156/data/negative1K_500_0.5_90.27neg_spu_75.beton" "/mnt/lustre/work/oh/owl156/data/negative_paths_log/27neg_spu_75.log" "/mnt/lustre/work/oh/owl156/data/negative_paths_log/27neg_spu_50.log"