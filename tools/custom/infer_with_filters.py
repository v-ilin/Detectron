from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sympy import *
import argparse
import cv2
import glob
import logging
import os
import sys

import numpy as np
from skimage import io
from skimage.color import rgb2hed, rgb2gray
from skimage.exposure import rescale_intensity


from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

person_class_index = 1
hardhat_class_index = 2

CHANCE_THRESHOLD = 0.7

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
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


def find_intruders(persons, hardhats):
    per_pols = [create_polygon(p) for p in persons]
    hat_pols = [create_polygon(h) for h in hardhats]

    pers_without_hats = []

    # for per_pol in per_pols:
    #     per_with_hat = False
    #     for hat_pol in hat_pols:
    #         per_with_hat = per_with_hat or len(
    #             per_pol.intersection(hat_pol)) != 0
    #     if not per_with_hat:
    #         pers_without_hats.append(per_pol)

    for p in range(len(persons)):
        per_with_hat = False
        for h in range(len(hardhats)):
            per_with_hat = per_with_hat or len(
                per_pols[p].intersection(hat_pols[h])) != 0
        if not per_with_hat:
            pers_without_hats.append(persons[p])

    return pers_without_hats


def get_class_probability(bbox_coord):
    return bbox_coord[4]

def recognize_image(im_name):


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    im_list = list(im_list)
    im_list.sort()

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )

        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)

        with c2_utils.NamedCudaScope(0):
            cls_boxes, _, _ = infer_engine.im_detect_all(model, im, None)

        boxes, _, _, classes = vis_utils.convert_from_cls_format(
            cls_boxes, None, None)

        persons = []
        hardhats = []

        for person in cls_boxes[person_class_index]:
            if person[4] > CHANCE_THRESHOLD:
                persons.append(person)

        for hardhat in cls_boxes[hardhat_class_index]:
            if hardhat[4] > CHANCE_THRESHOLD:
                hardhats.append(hardhat)

        print("\n")
        print("persons: " + str(persons))

        print("\n")
        print("hardhats: " + str(hardhats))

        intruder = find_intruders(persons, hardhats)

        if len(intruder) != 0:
            print("violation " + im_name)

            vis_utils.vis_one_image(
                im[:, :, ::-1],  # BGR -> RGB for visualization
                im_name,
                args.output_dir,
                cls_boxes,
                None,
                None,
                dataset=dummy_coco_dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2
            )


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    print("\n")
    print(args)
    main(args)
    print("\n")
