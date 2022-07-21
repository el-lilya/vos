# ./detection/evaluation/apply_net.sh
DEVICE=5
RANDOMSEED=0
SAVEFIGDIR='/netapp/l.lemikhova//projects/VOS_forked/vos/savefig/'

CONFIG=Fruits-Detection/vanilla_1.yaml
VISUALIZE=1
ENERGY_THR=-100

# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_id_val  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_id_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_sim_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_diff_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_sim_val  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_diff_val  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_sim_train  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_diff_train  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR


# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset coco_id_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset coco_ood_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset coco_ood_train  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset coco_ood_neg_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR

# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset deep_fruits_id_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset deep_fruits_ood_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR

CONFIG=Fruits-Detection/vos_1.yaml
VISUALIZE=0
ENERGY_THR=6.57
# ENERGY_THR=-100

# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_id_val  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_id_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_sim_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_diff_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_sim_val  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_diff_val  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_sim_train  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_diff_train  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR

# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset coco_id_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset coco_ood_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset coco_ood_train  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset coco_ood_neg_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR

# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset deep_fruits_id_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset deep_fruits_ood_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR --energy_thr $ENERGY_THR

