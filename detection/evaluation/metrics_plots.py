# python fruits_coco_plot.py --name vos_7 --thres 0.5062 --energy 1 --seed 0

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

recall_levels_default = [0.8, 0.95]


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--energy', type=int, default=1, help='noise for Odin')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--thres', default=1., type=float)
parser.add_argument('--name', default='vanilla_1', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model', default='faster-rcnn', type=str)
parser.add_argument('--id_dataset', default='deep_fruits_id_test', type=str)
parser.add_argument('--ood_dataset', default='deep_fruits_ood_test', type=str)
args = parser.parse_args()



concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

root = '/netapp/l.lemikhova/projects/VOS_forked/vos/detection/data/configs/Fruits-Detection/'

id_data = pickle.load(open(root +args.name+'/random_seed'+'_'+str(args.seed)+f'/inference/{args.id_dataset}/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
ood_data = pickle.load(open(root+args.name+'/random_seed'+'_'+str(args.seed)+f'/inference/{args.ood_dataset}/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))

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
print(f'{args.name} for {args.ood_dataset}:')

auroc, aupr, fprs = get_measures(-id_score, -ood_score, recall_levels=recall_levels_default, name=f'{args.name}_{args.ood_dataset}', plot=True)
print_measures(auroc, aupr, fprs, 'energy', recall_levels=recall_levels_default)

print('len(id_score)', len(id_score))
print('len(ood_score)', len(ood_score))
# print(f'number of false positives: ')
# print(f'{recall_levels_default[0]}: {len(ood_score) * fprs[0]}')
# print(f'{recall_levels_default[1]}: {len(ood_score) * fprs[1]}')
print('-'*10)
