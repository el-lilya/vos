import os

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Project imports
import core.datasets.metadata as metadata


def setup_fruits_dataset(img_dir, ann_dir, dataset_name, stages):
    ann_dir = f'{ann_dir}/{dataset_name}'
    for stage in stages:
        dataset_name_stage = f'{dataset_name}_{stage}'
        json_annotations = os.path.join(ann_dir, 'COCO-Format', f'{stage}_coco_format.json')
        register_coco_instances(
            dataset_name_stage,
            {},
            json_annotations,
            img_dir)
        MetadataCatalog.get(
            dataset_name_stage).thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
        MetadataCatalog.get(
            dataset_name_stage).thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID


def setup_all_datasets(dataset_dir=None, image_root_corruption_prefix=None):
    """
    Registers all datasets as instances from COCO

    Args:
        dataset_dir(str): path to dataset directory

    """
    root_fruits_dir='/ssd/l.lemikhova/data/fruits'
    root_coco_all_dir='/ssd/l.lemikhova/data/animals/coco/full'
    root_ann_dir='/netapp/l.lemikhova/projects/VOS_forked/vos/data/'
    # coco_id
    setup_fruits_dataset(
        img_dir=f'{root_fruits_dir}/COCO/images/',
        ann_dir=f'{root_ann_dir}/COCO/annotation',
        dataset_name='coco_id',
        stages=['train', 'val', 'test']
    )
    # coco_ood
    setup_fruits_dataset(
        img_dir=f'{root_fruits_dir}/COCO/images/',
        ann_dir=f'{root_ann_dir}/COCO/annotation',
        dataset_name='coco_ood',
        stages=['train', 'val', 'test']
    )
    # coco_ood_neg
    setup_fruits_dataset(
        img_dir=root_coco_all_dir,
        ann_dir=f'{root_ann_dir}/COCO/annotation',
        dataset_name='coco_ood_neg',
        stages=['test']
    )
    # openim_id
    setup_fruits_dataset(
        img_dir=f'{root_fruits_dir}/OpenIm/images/',
        ann_dir=f'{root_ann_dir}/OpenIm/annotation',
        dataset_name='openim_id',
        stages=['train', 'val', 'test']
    )
    # openim_ood
    setup_fruits_dataset(
        img_dir=f'{root_fruits_dir}/OpenIm/images/',
        ann_dir=f'{root_ann_dir}/OpenIm/annotation',
        dataset_name='openim_ood',
        stages=['train', 'val', 'test']
    )
    # openim_ood_sim
    setup_fruits_dataset(
        img_dir=f'{root_fruits_dir}/OpenIm/images/',
        ann_dir=f'{root_ann_dir}/OpenIm/annotation',
        dataset_name='openim_ood_sim',
        stages=['train', 'val', 'test']
    )
    # openim_ood_diff
    setup_fruits_dataset(
        img_dir=f'{root_fruits_dir}/OpenIm/images/',
        ann_dir=f'{root_ann_dir}/OpenIm/annotation',
        dataset_name='openim_ood_diff',
        stages=['train', 'val', 'test']
    )
    # deep_fruits_id
    setup_fruits_dataset(
        img_dir=f'{root_fruits_dir}/DeepFruits/images/',
        ann_dir=f'{root_ann_dir}/DeepFruits/annotation',
        dataset_name='deep_fruits_id',
        stages=['test']
    )
    # deep_fruits_ood
    setup_fruits_dataset(
        img_dir=f'{root_fruits_dir}/DeepFruits/images/',
        ann_dir=f'{root_ann_dir}/DeepFruits/annotation',
        dataset_name='deep_fruits_ood',
        stages=['test']
    )
