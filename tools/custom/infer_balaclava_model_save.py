#!/usr/bin/env python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sympy import *
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
import numpy as np
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

person_class_index = 1
hardhat_class_index = 2
not_hardhat_class_index = 3

CHANCE_THRESHOLD = 0.7
BALACLAVA_THRESHOLD = 50

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

hardhat_model_path = "/home/user/vilin/detectron-output/hardhats_persons_4/train/hardhats_persons_4_train" \
                     "/generalized_rcnn/model_final.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str)
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str)
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str)
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str)
    parser.add_argument(
        '--csv-path',
        dest='csv_path',
        help='path to csv file',
        default='/tmp/data.csv',
        type=str)
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def create_polygon(cords):
    left_top = (cords[0], cords[1])
    left_bottom = (cords[0], cords[3])
    right_top = (cords[2], cords[1])
    right_bottom = (cords[2], cords[3])
    return Polygon(left_bottom, left_top, right_top, right_bottom)


def intersects(person, hardhat):
    intersection = person.intersection(hardhat)
    return len(intersection) != 0 or person.encloses_point(hardhat.vertices[0])


def find_intruders(persons, hardhats):
    per_pols = [create_polygon(p) for p in persons]
    hat_pols = [create_polygon(h) for h in hardhats]

    pers_without_hats = []

    for p in range(len(persons)):
        per_with_hat = False
        for h in range(len(hardhats)):
            per_with_hat = per_with_hat or intersects(per_pols[p], hat_pols[h])
        if not per_with_hat:
            pers_without_hats.append(persons[p])

    return pers_without_hats


def get_class_probability(bbox_coord):
    return bbox_coord[4]


def dist(x1, y1, x2, y2):
    return sqrt((y1 - y2)**2 + (x2 - x1)**2)


def is_balaclava(hh, image):
    hh = [int(h) for h in hh]
    x1, y1, x2, y2 = hh[0], hh[1], hh[2], hh[3]
    cx, cy = np.average([x1, x2]), np.average([y1, y2])
    d_max = dist(x1, y1, cx, cy)
    colors = []
    for i in range(y1, y2):
        for j in range(x1, x2):
            d = dist(j, i, cx, cy)
            m = (d_max - d) / d_max
            colors.extend([int(color * m) for color in image[i, j]])

    # cv2.imwrite("/tmp/balaclava.jpg", image[y1:y2, x1:x2])
    average_color = np.average(colors)
    print("Average is :" + str(average_color) + " when threshold is: " +
          str(BALACLAVA_THRESHOLD))
    return average_color < BALACLAVA_THRESHOLD


def main(args):
    logging.disable("warning")
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    im_list = list(im_list)
    im_list.sort()

    csv = open(args.csv_path, 'w+')

    for i, im_name in enumerate(im_list):
        # out_name = os.path.join(
        #     args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf'))

        im = cv2.imread(im_name)

        # logger.info('Processing {} -> {}'.format(im_name, out_name))

        people_model = infer_engine.initialize_model_from_cfg(args.weights)

        with c2_utils.NamedCudaScope(0):
            cls_boxes, _, _ = infer_engine.im_detect_all(
                people_model, im, None)

        hardhat_model = infer_engine.initialize_model_from_cfg(
            hardhat_model_path)

        with c2_utils.NamedCudaScope(0):
            hardhat_cls_boxes, _, _ = infer_engine.im_detect_all(
                hardhat_model, im, None)

        cls_boxes[hardhat_class_index] = hardhat_cls_boxes[hardhat_class_index]

        persons = []
        hardhats = []
        not_hardhats = []

        for person in cls_boxes[person_class_index]:
            if person[4] > CHANCE_THRESHOLD:
                persons.append(person)

        for hardhat in cls_boxes[hardhat_class_index]:
            if hardhat[4] > CHANCE_THRESHOLD:
                if not is_balaclava(hardhat, im):
                    hardhats.append(hardhat)
                else:
                    not_hardhats.append(hardhat)

        cls_boxes[person_class_index] = persons
        cls_boxes[hardhat_class_index] = hardhats
        cls_boxes.append(not_hardhats)

        # remove this if you want to save all files
        # if len(not_hardhats) == 0:
        #     continue

        file_name = os.path.split(im_name)[1]
        file_without_ext = os.path.splitext(file_name)[0]

        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            file_without_ext,
            args.output_dir,
            cls_boxes,
            None,
            None,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2)

    csv.close()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    # print(args)
    main(args)
