# ./detection/evaluation/plots.sh
DEVICE=2
# THRES=0.5382
THRES=0.5602
NAME=vos_from_scratch_7

CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/openim_plot.py --name $NAME --thres $THRES --energy 1 --seed 0
CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/deep_fruits_plot.py --name $NAME --thres $THRES --energy 1 --seed 0

THRES=0.5638
NAME=vanilla_from_scratch_7

CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/openim_plot.py --name $NAME --thres $THRES --energy 1 --seed 0
CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/deep_fruits_plot.py --name $NAME --thres $THRES --energy 1 --seed 0