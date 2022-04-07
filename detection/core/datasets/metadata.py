from collections import ChainMap

# Detectron imports
from detectron2.data import MetadataCatalog


COCO_FRUITS_THING_CLASSES = ['banana', 'apple', 'orange']

COCO_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i: i} for i in range(len(COCO_FRUITS_THING_CLASSES))]))
    # ChainMap(*[{i + 1: i} for i in range(len(COCO_FRUITS_THING_CLASSES))]))

OPENIM_FRUITS_THING_CLASSES = ['banana', 'apple', 'orange', 'strawberry', 'tomato', 'lemon', 'pear', 'grape', 'watermelon', 'pineapple', 'pomegranate', 'grapefruit', 'peach', 'mango', 'common fig', 'cantaloupe']
OPENIM_FRUITS_ID_THING_CLASSES = OPENIM_FRUITS_THING_CLASSES[:7]
OPENIM_FRUITS_OOD_THING_CLASSES = OPENIM_FRUITS_THING_CLASSES[7:]

OPENIM_FRUITS_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i: i} for i in range(len(OPENIM_FRUITS_ID_THING_CLASSES))]))
    # ChainMap(*[{i + 1: i} for i in range(len(OPENIM_FRUITS_THING_CLASSES))]))