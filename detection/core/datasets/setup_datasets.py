import os

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Project imports
import core.datasets.metadata as metadata


def setup_all_datasets(dataset_dir, image_root_corruption_prefix=None,):
    """
    Registers all datasets as instances from COCO

    Args:
        dataset_dir(str): path to dataset directory

    """
    root_img_dir='/ssd/l.lemikhova/data/fruits'
    root_ann_dir='/netapp/l.lemikhova/projects/VOS_forked/vos/data'
    setup_coco_only_fruits_dataset(root_img_dir, root_ann_dir)
    setup_coco_ext_fruits_dataset(root_img_dir, root_ann_dir)
    setup_openim_id_fruits_dataset(root_img_dir, root_ann_dir)
    setup_openim_ood_fruits_dataset(root_img_dir, root_ann_dir)
    setup_openim_ood_sim_fruits_dataset(root_img_dir, root_ann_dir)
    setup_openim_ood_diff_fruits_dataset(root_img_dir, root_ann_dir)
    setup_deep_fruits_id_fruits_dataset(root_img_dir, root_ann_dir)
    setup_deep_fruits_ood_fruits_dataset(root_img_dir, root_ann_dir)
    setup_openim_full_fruits_dataset(root_img_dir, root_ann_dir)

def setup_coco_only_fruits_dataset(root_img_dir='/ssd/l.lemikhova/data/fruits', root_ann_dir='./data'):
    img_dir = f'{root_img_dir}/coco'
    train_image_dir = os.path.join(img_dir, 'train2017')
    test_image_dir = os.path.join(img_dir, 'train2017')

    ann_dir = f'{root_ann_dir}/coco_only'
    train_json_annotations = os.path.join(
        ann_dir, 'COCO-Format', 'train_coco_format.json')
    test_json_annotations = os.path.join(
        ann_dir, 'COCO-Format', 'val_coco_format.json')
    register_coco_instances(
        "coco_fruits_train",
        {},
        train_json_annotations,
        train_image_dir)
    # del MetadataCatalog.get("coco_fruits_train").thing_dataset_id_to_contiguous_id
    MetadataCatalog.get(
        "coco_fruits_train").thing_classes = metadata.COCO_FRUITS_THING_CLASSES
    MetadataCatalog.get(
        "coco_fruits_train").thing_dataset_id_to_contiguous_id = metadata.COCO_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "coco_fruits_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_fruits_val").thing_classes = metadata.COCO_FRUITS_THING_CLASSES
    MetadataCatalog.get(
        "coco_fruits_val").thing_dataset_id_to_contiguous_id = metadata.COCO_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_coco_ext_fruits_dataset(root_img_dir='/ssd/l.lemikhova/data/fruits', root_ann_dir='./data'):
    img_dir = f'{root_img_dir}/coco'
    train_image_dir = os.path.join(img_dir, 'train2017')
    test_image_dir = os.path.join(img_dir, 'train2017')

    ann_dir = f'{root_ann_dir}/coco_ext'
    train_json_annotations = os.path.join(
        ann_dir, 'COCO-Format', 'train_coco_format.json')
    test_json_annotations = os.path.join(
        ann_dir, 'COCO-Format', 'val_coco_format.json')
    register_coco_instances(
        "coco_ext_fruits_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "coco_ext_fruits_train").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    print(MetadataCatalog.get("coco_ext_fruits_train").thing_classes)
    MetadataCatalog.get(
        "coco_ext_fruits_train").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "coco_ext_fruits_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_ext_fruits_val").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "coco_ext_fruits_val").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_openim_id_fruits_dataset(root_img_dir='/ssd/l.lemikhova/data/fruits', root_ann_dir='./data'):
    img_dir = f'{root_img_dir}/openim'
    train_image_dir = img_dir
    test_image_dir = img_dir
    ann_dir = f'{root_ann_dir}/openim_id'

    train_json_annotations = os.path.join(ann_dir, 'COCO-Format', 'train_coco_format.json')
    val_json_annotations = os.path.join(ann_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(ann_dir, 'COCO-Format', 'test_coco_format.json')

    register_coco_instances(
        "openim_id_fruits_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "openim_id_fruits_train").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "openim_id_fruits_train").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "openim_id_fruits_val",
        {},
        val_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "openim_id_fruits_val").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "openim_id_fruits_val").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "openim_id_fruits_test",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openim_id_fruits_test").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "openim_id_fruits_test").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_openim_ood_fruits_dataset(root_img_dir='/ssd/l.lemikhova/data/fruits', root_ann_dir='./data'):
    img_dir = f'{root_img_dir}/openim'
    test_image_dir = img_dir
    ann_dir = f'{root_ann_dir}/openim_ood'
    val_json_annotations = os.path.join(ann_dir, 'COCO-Format', 'val_coco_format.json')
    test_json_annotations = os.path.join(ann_dir, 'COCO-Format', 'test_coco_format.json')

    register_coco_instances(
        "openim_ood_fruits_val",
        {},
        val_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openim_ood_fruits_val").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "openim_ood_fruits_val").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "openim_ood_fruits_test",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openim_ood_fruits_test").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "openim_ood_fruits_test").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID
        
def setup_openim_full_fruits_dataset(root_img_dir='/ssd/l.lemikhova/data/fruits', root_ann_dir='./data'):
    img_dir = f'{root_img_dir}/openim'
    train_image_dir = img_dir
    test_image_dir = img_dir
    ann_dir = f'{root_ann_dir}/openim_full'
    train_json_annotations = os.path.join(
        ann_dir, 'COCO-Format', 'train_coco_format.json')
    test_json_annotations = os.path.join(
        ann_dir, 'COCO-Format', 'val_coco_format.json')
    register_coco_instances(
        "openim_full_fruits_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "openim_full_fruits_train").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "openim_full_fruits_train").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "openim_full_fruits_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openim_full_fruits_val").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "openim_full_fruits_val").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_deep_fruits_id_fruits_dataset(root_img_dir='/ssd/l.lemikhova/data/fruits', root_ann_dir='./data'):
    img_dir = f'{root_img_dir}/deep_fruits'
    test_image_dir = img_dir
    ann_dir = f'{root_ann_dir}/deep_fruits_id'
    
    test_json_annotations = os.path.join(
        ann_dir, 'COCO-Format', 'test_coco_format.json')
    
    register_coco_instances(
        "deep_fruits_id_fruits_test",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "deep_fruits_id_fruits_test").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "deep_fruits_id_fruits_test").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_deep_fruits_ood_fruits_dataset(root_img_dir='/ssd/l.lemikhova/data/fruits', root_ann_dir='./data'):
    img_dir = f'{root_img_dir}/deep_fruits'
    test_image_dir = img_dir
    ann_dir = f'{root_ann_dir}/deep_fruits_ood'
    
    test_json_annotations = os.path.join(
        ann_dir, 'COCO-Format', 'test_coco_format.json')
    
    register_coco_instances(
        "deep_fruits_ood_fruits_test",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "deep_fruits_ood_fruits_test").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "deep_fruits_ood_fruits_test").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_openim_ood_sim_fruits_dataset(root_img_dir='/ssd/l.lemikhova/data/fruits', root_ann_dir='./data'):
    img_dir = f'{root_img_dir}/openim'
    test_image_dir = img_dir
    ann_dir = f'{root_ann_dir}/openim_ood_sim'
    
    test_json_annotations = os.path.join(
        ann_dir, 'COCO-Format', 'train_coco_format.json')
    
    register_coco_instances(
        "openim_ood_sim_fruits_test",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openim_ood_sim_fruits_test").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "openim_ood_sim_fruits_test").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_openim_ood_diff_fruits_dataset(root_img_dir='/ssd/l.lemikhova/data/fruits', root_ann_dir='./data'):
    img_dir = f'{root_img_dir}/openim'
    test_image_dir = img_dir
    ann_dir = f'{root_ann_dir}/openim_ood_diff'
    
    test_json_annotations = os.path.join(
        ann_dir, 'COCO-Format', 'train_coco_format.json')
    
    register_coco_instances(
        "openim_ood_diff_fruits_test",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openim_ood_diff_fruits_test").thing_classes = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    MetadataCatalog.get(
        "openim_ood_diff_fruits_test").thing_dataset_id_to_contiguous_id = metadata.OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID