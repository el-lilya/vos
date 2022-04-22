# python fruits_openim_plot.py --name vos_7 --thres 0.5062 --energy 1 --seed 0
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

def get_results(id_data, ood_data, args, type_ood):
    id_score = []
    ood_score = []

    if args.energy:
        id_score = -args.T * torch.logsumexp(torch.stack(id_data['inter_feat'])[:, :-1] / args.T, dim=1).cpu().data.numpy()
        ood_score = -args.T * torch.logsumexp(torch.stack(ood_data['inter_feat'])[:, :-1] / args.T, dim=1).cpu().data.numpy()
    else:
        id_score = -np.max(F.softmax(torch.stack(id_data['inter_feat'])[:, :-1], dim=1).cpu().data.numpy(), axis=1)
        ood_score = -np.max(F.softmax(torch.stack(ood_data['inter_feat'])[:, :-1], dim=1).cpu().data.numpy(), axis=1)

    
    print('len(id_score)', len(id_score))
    print('len(ood_score)', len(ood_score))

    measures = get_measures(-id_score, -ood_score, recall_level=recall_level_default, plot=True)
    if args.energy:
        print_measures(measures[0], measures[1], measures[2], 'energy')
    else:
        print_measures(measures[0], measures[1], measures[2], 'msp')

root = '/netapp/l.lemikhova/projects/VOS_forked/vos/detection/data/Faster-RCNN/coco_openim/'
print('*'*10)
print(args.name)
print('*'*10)
for type_ood in ['sim', 'diff']:
    print(f'Open images {type_ood}:')
    id_data = pickle.load(open(root +args.name+'/random_seed'+'_'+str(args.seed)+'/inference/openim_id_fruits_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
    ood_data = pickle.load(open(root+args.name+'/random_seed'+'_'+str(args.seed)+f'/inference/openim_ood_{type_ood}_fruits_test/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
    get_results(id_data, ood_data, args, type_ood=type_ood)
    print('-'*10)