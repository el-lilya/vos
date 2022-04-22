# ./detection/evaluation/plots.sh
DEVICE=2

# THRES=0.5602
THRES=0.4564
NAME=vos_from_scratch_7
RANDOMSEED=0
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/deep_fruits_plot.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED
CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/coco_fruits_plot.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/openim_plot.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED

# # THRES=0.5638
# THRES=0.4425
# NAME=vanilla_from_scratch_7
# RANDOMSEED=0
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/deep_fruits_plot.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED
# # CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/openim_plot.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED

# THRES=0.5387
# RANDOMSEED=1
# # THRES=0.5062
# # RANDOMSEED=0
# NAME=vos_7

# CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/openim_plot.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/deep_fruits_plot.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED
