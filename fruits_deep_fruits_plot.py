# python fruits_coco_plot.py --name vos_7 --thres 0.5062 --energy 1 --seed 0
# python fruits_coco_plot.py --name vanilla_from_scratch_7 --thres 0.5701 --energy 1 --seed 0
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from metric_utils import *

# recall_level_default = 0.8
recall_level_default = 0.95


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--energy', type=int, default=1, help='noise for Odin')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--thres', default=1., type=float)
parser.add_argument('--name', default=1., type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model', default='faster-rcnn', type=str)
args = parser.parse_args()



concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


# detection/data/Faster-RCNN/coco_openim/vos_7/random_seed_0/inference/openim_id_fruits_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_0.5062.pkl
# detection/data/Fruits-Detection/Faster-RCNN/coco_openim/vos_7/random_seed_0/inference/openim_id_fruits_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_0.5698.pkl
root = '/netapp/l.lemikhova/projects/VOS_forked/vos/detection/data/Faster-RCNN/coco_openim/'

# id_data = pickle.load(open(root +args.name+'/random_seed'+'_'+str(args.seed)+'/inference/openim_id_fruits_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
# ood_data = pickle.load(open(root+args.name+'/random_seed'+'_'+str(args.seed)+'/inference/openim_ood_fruits_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))

id_data = pickle.load(open(root +args.name+'/random_seed'+'_'+str(args.seed)+'/inference/deep_fruits_id_fruits_test/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
ood_data = pickle.load(open(root+args.name+'/random_seed'+'_'+str(args.seed)+'/inference/deep_fruits_ood_fruits_test/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))

id = 0
T = 1
id_score = []
ood_score = []



if args.energy:
    id_score = -args.T * torch.logsumexp(torch.stack(id_data['inter_feat'])[:, :-1] / args.T, dim=1).cpu().data.numpy()
    ood_score = -args.T * torch.logsumexp(torch.stack(ood_data['inter_feat'])[:, :-1] / args.T, dim=1).cpu().data.numpy()
else:
    id_score = -np.max(F.softmax(torch.stack(id_data['inter_feat'])[:, :-1], dim=1).cpu().data.numpy(), axis=1)
    ood_score = -np.max(F.softmax(torch.stack(ood_data['inter_feat'])[:, :-1], dim=1).cpu().data.numpy(), axis=1)

###########
########
print(len(id_score))
print(len(ood_score))

measures = get_measures(-id_score, -ood_score, recall_level=recall_level_default, plot=True)
if args.energy:
    print_measures(measures[0], measures[1], measures[2], 'energy')
else:
    print_measures(measures[0], measures[1], measures[2], 'msp')

