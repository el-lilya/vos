# ./detection/evaluation/apply_net_no_fig.sh
DEVICE=2
VISUALIZE=0
SAVEFIGDIR='./evaluation/'
CONFIG=Fruits-Detection/Faster-RCNN/coco_openim/vos_from_scratch_7.yaml
RANDOMSEED=0

CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_id_fruits_val  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_id_fruits_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_sim_fruits_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_diff_fruits_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR

CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset deep_fruits_id_fruits_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset deep_fruits_ood_fruits_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR


CONFIG=Fruits-Detection/Faster-RCNN/coco_openim/vanilla_from_scratch_7.yaml
RANDOMSEED=0

CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_id_fruits_val  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR
# CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_id_fruits_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_sim_fruits_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset openim_ood_diff_fruits_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR

CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset deep_fruits_id_fruits_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR
CUDA_VISIBLE_DEVICES=$DEVICE python detection/apply_net.py --dataset-dir data --test-dataset deep_fruits_ood_fruits_test  --config-file $CONFIG --inference-config Inference/standard_nms.yaml  --random-seed $RANDOMSEED --image-corruption-level 0 --visualize $VISUALIZE --savefigdir $SAVEFIGDIR