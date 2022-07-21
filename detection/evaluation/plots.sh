# ./detection/evaluation/plots.sh
RANDOMSEED=0
DEVICE=5

# ID_DATASET='deep_fruits_id_test'
# OOD_DATASET='deep_fruits_ood_test'
# THRES=0.5686
# NAME=vanilla_1
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET
# THRES=0.5745
# NAME=vos_1
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET


# ID_DATASET='coco_id_test'
# OOD_DATASET='coco_ood_neg_test'
# THRES=0.5686
# NAME=vanilla_1
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET
# THRES=0.5745
# NAME=vos_1
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET


ID_DATASET='openim_id_test'
OOD_DATASET='openim_ood_sim_test'
THRES=0.5686
NAME=vanilla_1
CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET
THRES=0.5745
NAME=vos_1
CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET

ID_DATASET='openim_id_test'
OOD_DATASET='openim_ood_diff_test'

THRES=0.5686
NAME=vanilla_1
CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET
THRES=0.5745
NAME=vos_1
CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET

ID_DATASET='openim_id_test'
OOD_DATASET='openim_ood_sim_val'
THRES=0.5686
NAME=vanilla_1
CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET
THRES=0.5745
NAME=vos_1
CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET

ID_DATASET='openim_id_test'
OOD_DATASET='openim_ood_diff_val'

THRES=0.5686
NAME=vanilla_1
CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET
THRES=0.5745
NAME=vos_1
CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET


# ID_DATASET='openim_id_test'
# OOD_DATASET='openim_ood_test'
# THRES=0.5686
# NAME=vanilla_1
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET
# THRES=0.5745
# NAME=vos_1
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/evaluation/metrics_plots.py --name $NAME --thres $THRES --energy 1 --seed $RANDOMSEED --id_dataset $ID_DATASET --ood_dataset $OOD_DATASET

python savefig/plots.py