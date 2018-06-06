#!/usr/bin/env python2

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from pprint import pprint
from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
from sympy import *
from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

CHANCE_THRESHOLD = 0.7
person_class_index = 1
left_ankle_class_index = 15
right_ankle_class_index = 16
# x1, y1, x2, y2
danger_zone = [136, 453, 587, 894]


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


def is_ankle_in_danger_zone(ankle, zone):
    return zone.encloses_point(ankle)


def check_ankles_in_danger_zone(ankles, zone):
    ankles_p = [Point2D(a[0], a[1]) for a in ankles]
    danger_zone_p = create_polygon(zone)
    for a in ankles_p:
        if is_ankle_in_danger_zone(a, danger_zone_p):
            return True
    return False


def collect_ankles(cls_keyps):
    ankles = []
    for person in cls_keyps:
        x_keyps = person[0]
        y_keyps = person[1]
        left_ankle = [x_keyps[left_ankle_class_index],
                      y_keyps[left_ankle_class_index]]
        right_ankle = [x_keyps[right_ankle_class_index],
                       y_keyps[right_ankle_class_index]]
        ankles.append(left_ankle)
        ankles.append(right_ankle)

    return ankles


def draw_danger_zone(frame, danger_zone):
    x1 = danger_zone[0]
    y1 = danger_zone[1]
    x2 = danger_zone[2]
    y2 = danger_zone[3]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, _, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )

        persons = []
        person_keys = []

        for p in range(len(cls_boxes[person_class_index])):
            person = cls_boxes[person_class_index][p]
            person_key = cls_keyps[person_class_index][p]
            if person[4] > CHANCE_THRESHOLD:
                persons.append(person)
                person_keys.append(person_key)

        if len(persons) == 0:
            continue

        cls_boxes[person_class_index] = persons
        cls_keyps[person_class_index] = person_keys

        pprint(len(persons))

        pprint(len(cls_boxes[person_class_index]))

        ankles = collect_ankles(cls_keyps[person_class_index])

        print(str(ankles))

        if not check_ankles_in_danger_zone(ankles, danger_zone):
            continue

        print("violation")

        frame = vis_utils.vis_one_image_opencv(
            im,
            cls_boxes,
            None,
            cls_keyps,
            thresh=0.6,
            show_box=True,
            dataset=dummy_coco_dataset,
            show_class=True)

        draw_danger_zone(frame, danger_zone)
        print(im_name + '.jpg')
        cv2.imwrite(im_name + '.jpg', frame)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
