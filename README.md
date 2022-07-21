# Fruits detection with VOS

This is the source code for part of my Ms. Thesis for Out-Of-Distribution Fruits Detection with VOS method introduced in the paper [***VOS: Learning What You Don’t Know by Virtual Outlier Synthesis***](https://openreview.net/forum?id=TW7d65uYu5M) by Xuefeng Du, Zhaoning Wang, Mu Cai, and Yixuan Li

## Requirements
```
pip install -r requirements.txt
```
In addition, install detectron2 following [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Dataset Preparation

Datasets were prepaired to be in standard COCO format. They were formed from images containing fruits from COCO, Open Images and Deep Fruits datasets.

## Training

Firstly, enter the detection folder by running
```
cd detection
```

Before training, modify the dataset address by changing "dataset-dir" according to your local dataset address.

**Vanilla**
```
python train_net.py 
--dataset-dir path/to/dataset/dir
--num-gpus 8 
--config-file configs/Fruits-Detection/vanilla_1.yaml
--random-seed 0 
--resume
```
**VOS**
```
python train_net_gmm.py 
--dataset-dir path/to/dataset/dir
--num-gpus 8 
--config-file configs/Fruits-Detection/vos_1.yaml
--random-seed 0 
--resume
```

Before training on VOS, change "VOS.STARTING_ITER" and "VOS.SAMPLE_NUMBER" in the config file to the desired numbers in paper.

## Evaluation

Firstly run on the in-distribution dataset, then run on the OOD dataset. You can find the code for evaluation on the used datasets here:
```
./detection/evaluation/apply_net.sh
```
Obtain the metrics using:
```
./detection/evaluation/plots.sh
```
Here the threshold is determined according to [ProbDet](https://github.com/asharakeh/probdet). It will be displayed in the screen as you finish evaluating on the in-distribution dataset.


