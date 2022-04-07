# python vos/detection/core/datasets/convert_openim_fruits_ood_to_coco.py --dataset-dir data/openim_ood
import argparse
import csv
import cv2
import json
import os
import metadata

from tqdm import tqdm


def get_openim_fruits_ood_dicts(stage, root = "./data/openim", output_dir=None):  #'/content/coco'
    openim_classes_need = metadata.OPENIM_FRUITS_ID_THING_CLASSES
    category_mapper = dict(zip(openim_classes_need, range(len(openim_classes_need))))
    # category_mapper = {'banana': 0, 'apple': 1, 'orange': 2}
    images_list = []
    annotations_list = []    
    f = open(f'{root}/annotation/fruits_{stage}.txt')
    filename = str.rstrip(f.readline()) # 'The first line.\n'
    idx = 0
    while filename != '':
        num_ID_bboxes = 0
        full_filename = os.path.join(root, filename)
        height, width = cv2.imread(full_filename).shape[:2]
        num = int(f.readline())
        for i in range(num):
            line = f.readline()
            category_name = line.split(' ')[-1][:-1]
            if category_name in category_mapper.keys():
                num_ID_bboxes += 1
        if  num_ID_bboxes == 0:
            images_list.append({'id': idx,
                                'width': width,
                                'height': height,
                                'file_name': filename,
                                'license': 1})
        filename = str.rstrip(f.readline()) # 'The first line.\n'
        idx += 1
    print(len(images_list))
    categories = [{"supercategory": "food", "id": i, "name": classname} for classname, i in category_mapper.items()]
    licenses = [{'id': 1,
                'name': 'none',
                'url': 'none'}]
    json_dict = {'info': {'year': 2020},
                'licenses': licenses,
                'categories': categories,
                'images': images_list,
                'annotations': annotations_list}

    file_name = f'{output_dir}/{stage}_coco_format.json'
    # file_name = os.path.join(output_dir, f'{stage}_coco_format.json')
    with open(file_name, 'w') as outfile:
        json.dump(json_dict, outfile)

    return 0


def main(args):
    dataset_dir = args.dataset_dir

    if args.output_dir is None:
        output_dir = os.path.expanduser(
            os.path.join(dataset_dir, 'COCO-Format'))
    else:
        output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)
    for stage in tqdm(['test', 'val', 'train']):
        get_openim_fruits_ood_dicts(stage, output_dir=output_dir)
    

if __name__ == "__main__":
    # Create arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=str)

    parser.add_argument(
        "--output-dir",
        required=False,
        type=str,
        help='converted dataset write directory')

    args = parser.parse_args()
    main(args)
