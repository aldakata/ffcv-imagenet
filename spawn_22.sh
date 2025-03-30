mu=9
lmbd=0
num_classes=26
validationset=/mnt/lustre/work/oh/owl156/data/ImageNet26_22more_val_400_0.5_90.beton
opt=adam
# paths=(22more_spurious_400_0.5_90_5.beton 22more_spurious_400_0.5_90_15.beton 22more_spurious_400_0.5_90_50.beton )
# name=22_SPURIOUS
# for i in {0..2}
# do
#     for p in "${paths[@]}"
#     do
#         sbatch train_4.sbatch $name $opt /mnt/lustre/work/oh/owl156/data/$p 0 $validationset $num_classes $mu $lmbd
#     done
# done

# paths=(22more_random_400_0.5_90_5.beton 22more_random_400_0.5_90_15.beton 22more_random_400_0.5_90_50.beton )
# name=22_RANDOM
# for i in {0..2}
# do
#     for p in "${paths[@]}"
#     do
#         sbatch train_4.sbatch $name $opt /mnt/lustre/work/oh/owl156/data/$p 0 $validationset $num_classes $mu $lmbd
#     done
# done

paths=(22more_neg_400_0.5_90_5.beton 22more_neg_400_0.5_90_15.beton)
# paths=(22more_neg_400_0.5_90_50.beton )
name=22_NEGATIVE
for i in {0..2}
do
    for p in "${paths[@]}"
    do
        sbatch train_4.sbatch $name $opt /mnt/lustre/work/oh/owl156/data/$p 0 $validationset $num_classes $mu $lmbd
    done
done

# paths=(22more_pos_400_0.5_90.beton)
# name=22_POSITIVE
# for i in {0..2}
# do
#     for p in "${paths[@]}"
#     do
#         sbatch train_4.sbatch $name $opt /mnt/lustre/work/oh/owl156/data/$p 0 $validationset $num_classes $mu $lmbd
#     done
# done