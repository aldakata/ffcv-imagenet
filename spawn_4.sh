# 4 MORE SPURIOUS
# paths=(4more_spurious_400_0.5_90_15.beton 4more_spurious_400_0.5_90_50.beton 4more_spurious_400_0.5_90_75.beton)
# for i in {0..4}
# do
#     for opt in sgd
#     do
#         for p in "${paths[@]}"
#         do
#             sbatch train_4.sbatch 4MORE_SPURIOUS $opt /mnt/lustre/work/oh/owl156/data/$p 0 /mnt/lustre/work/oh/owl156/data/ImageNet4_4more_val_400_0.5_90.beton 4
#         done
#     done
# done

# # 4 MORE RANDOM
# paths=(4more_random_400_0.5_90_15.beton 4more_random_400_0.5_90_50.beton 4more_random_400_0.5_90_150.beton)
# for i in {0..4}
# do
#     for opt in sgd
#     do
#         for p in "${paths[@]}"
#         do
#             sbatch train_4.sbatch 4MORE_RANDOM $opt /mnt/lustre/work/oh/owl156/data/$p 0 /mnt/lustre/work/oh/owl156/data/ImageNet4_4more_val_400_0.5_90.beton 4
#         done
#     done
# done

# # 4 SPURIOUS
# paths=(4_spurious_split_400_0.5_90_15.beton 4_spurious_split_400_0.5_90_50.beton)
# for i in {0..10}
# do
#     for opt in adam # sgd
#     do
#         for p in "${paths[@]}"
#         do
#             sbatch train_4.sbatch 4SPURIOUSSPLITGOOD $opt /mnt/lustre/work/oh/owl156/data/$p 0 /mnt/lustre/work/oh/owl156/data/ImageNet4_val_400_0.5_90.beton 4
#         done
#     done
# done

# 4 RANDOM
# paths=(4_random_400_0.5_90_15.beton 4_random_400_0.5_90_50.beton 4_random_400_0.5_90_150.beton)
# paths=(4_random_400_0.5_90_50.beton)

# for i in {0..4}
# do
#     for opt in adam
#     do
#         for p in "${paths[@]}"
#         do
#             sbatch train_4.sbatch 4RANDOMblur $opt /mnt/lustre/work/oh/owl156/data/$p 0 /mnt/lustre/work/oh/owl156/data/ImageNet4_val_400_0.5_90.beton 4
#         done
#     done
# done

# # 12 MORE SPURIOUS
# paths=(12more_spurious_400_0.5_90_15.beton 12more_spurious_400_0.5_90_50.beton 12more_spurious_400_0.5_90_75.beton)
# for i in {0..4}
# do
#     for opt in sgd
#     do
#         for p in "${paths[@]}"
#         do
#             sbatch train_4.sbatch 12MORE_SPURIOUS $opt /mnt/lustre/work/oh/owl156/data/$p 0 /mnt/lustre/work/oh/owl156/data/ImageNet4_8more_val_400_0.5_90.beton 12
#         done
#     done
# done

# # 12 MORE 150
# paths=(12more_400_0.5_90_150.beton)
# for i in {0..4}
# do
#     for opt in sgd
#     do
#         for p in "${paths[@]}"
#         do
#             sbatch train_4.sbatch 12MORE $opt /mnt/lustre/work/oh/owl156/data/$p 0 /mnt/lustre/work/oh/owl156/data/ImageNet4_8more_val_400_0.5_90.beton 12
#         done
#     done
# done

# # 12 MORE HARD
paths=(12FILTEREDHARD_400_0.5_90_5.beton 12FILTEREDHARD_400_0.5_90_15.beton 12FILTEREDHARD_400_0.5_90_50.beton)
for i in {0..4}
do
    for mu in 6 # 9
    do
        for opt in adam
        do
            for p in "${paths[@]}"
            do
                sbatch train_4.sbatch 12MOREHARD $opt /mnt/lustre/work/oh/owl156/data/$p 0 /mnt/lustre/work/oh/owl156/data/ImageNet4_8more_val_400_0.5_90.beton 12 $mu 1
            done
        done
    done
done

# # 12 MORE RANDOM
# paths=(12more_random_400_0.5_90_150.beton 12more_random_400_0.5_90_15.beton 12more_random_400_0.5_90_50.beton)
# for i in {0..4}
# do
#     for opt in adam
#     do
#         for p in "${paths[@]}"
#         do
#             sbatch train_4.sbatch 12MORE_RANDOM $opt /mnt/lustre/work/oh/owl156/data/$p 0 /mnt/lustre/work/oh/owl156/data/ImageNet4_8more_val_400_0.5_90.beton 12 6 0
#         done
#     done
# done

# paths=(4more_400_0.5_90_15.beton 4more_400_0.5_90_50.beton)
# for i in {0..19}
# do
#     for opt in sgd
#     do
#         for p in "${paths[@]}"
#         do
#             sbatch train_4.sbatch 4MORE $opt /mnt/lustre/work/oh/owl156/data/$p 0 /mnt/lustre/work/oh/owl156/data/ImageNet4_4more_val_400_0.5_90.beton 4
#         done
#     done
# done

# paths=(12more_400_0.5_90_15.beton 12more_400_0.5_90_50.beton 12more_400_0.5_90_150.beton)
# for i in {0..4}
# do
#     for opt in adam
#     do
#         for p in "${paths[@]}"
#         do
#             sbatch train_4.sbatch 12MOREGOOD $opt /mnt/lustre/work/oh/owl156/data/$p 0 /mnt/lustre/work/oh/owl156/data/ImageNet4_8more_val_400_0.5_90.beton 12 6 0
#         done
#     done
# done

# # 12 SPURIOUSSPLIT
# paths=(12more_spurious_split_400_0.5_90_5.beton 12more_spurious_split_400_0.5_90_15.beton 12more_spurious_split_400_0.5_90_50.beton)
# for i in {0..4}
# do
#     for opt in adam
#     do
#         for p in "${paths[@]}"
#         do
#             sbatch train_4.sbatch 12SPURIOUSSPLITGOOD $opt /mnt/lustre/work/oh/owl156/data/$p 0 /mnt/lustre/work/oh/owl156/data/ImageNet4_8more_val_400_0.5_90.beton 12 3 0
#         done
#     done
# done