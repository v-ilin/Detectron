#!/usr/bin/env python2

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

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

from pprint import pprint
from enum import IntEnum

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
        type=str)
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str)
    parser.add_argument(
        '--output-video',
        dest='output_video',
        help='output video file (/path/to/video.avi)',
        default=None,
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


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


Treshold = 0.6

person_history = []


class person_class_index(IntEnum):
    in_hardhat = 1
    without_hardhat = 4
    in_gloves = 3
    without_gloves = 7
    in_goggles = 5
    without_goggles = 6
    in_hood = 2


class person_hardhat_status(IntEnum):
    Undefined = 0
    In_Hardhat = 1
    Without_Hardhat = 2
    In_Hood = 3


class person:
    def __init__(self, bbox, status, gloves_on, goggles_on):
        self.bbox = bbox
        self.status = status
        self.gloves_on = gloves_on
        self.goggles_on = goggles_on


class hardhat_polygon:
    status = person_hardhat_status.Undefined

    def __init__(self, bbox, status):
        self.bbox = bbox
        self.status = status


class gloves_goggles_polygon:
    status = None

    def __init__(self, bbox, status):
        self.bbox = bbox
        self.status = status


def get_polygon_index_with_max_precision(bboxes):
    precitions = []

    for _, polygon in enumerate(bboxes):
        precitions.append(polygon.bbox[4])

    return int(max(precitions))


def convert_to_person(cls_boxes):
    current_person = person(None, None, None, None)

    hardhats_polygons = []

    if len(cls_boxes[int(person_class_index.in_hardhat)]) > 0:
        hardhats_polygons.append(hardhat_polygon(
            cls_boxes[int(person_class_index.in_hardhat)][0], person_hardhat_status.In_Hardhat))

    if len(cls_boxes[int(person_class_index.without_hardhat)]) > 0:
        hardhats_polygons.append(hardhat_polygon(
            cls_boxes[int(person_class_index.without_hardhat)][0], person_hardhat_status.Without_Hardhat))

    if len(cls_boxes[int(person_class_index.in_hood)]) > 0:
        hardhats_polygons.append(hardhat_polygon(
            cls_boxes[int(person_class_index.in_hood)][0], person_hardhat_status.In_Hood))

    polygons_count = len(hardhats_polygons)

    if polygons_count == 0:
        current_person.bbox = None
        current_person.status = None
    elif polygons_count == 1:
        current_person.bbox = hardhats_polygons[0].bbox
        current_person.status = hardhats_polygons[0].status
    else:
        max_precition_index = get_polygon_index_with_max_precision(
            hardhats_polygons)

        current_person.bbox = hardhats_polygons[max_precition_index].bbox
        current_person.status = hardhats_polygons[max_precition_index].status

    gloves_polygons = []

    if len(cls_boxes[int(person_class_index.in_gloves)]) > 0:
        gloves_polygons.append(gloves_goggles_polygon(
            cls_boxes[int(person_class_index.in_gloves)][0], True))

    if len(cls_boxes[int(person_class_index.without_gloves)]) > 0:
        gloves_polygons.append(gloves_goggles_polygon(
            cls_boxes[int(person_class_index.without_gloves)][0], False))

    gloves_polygons_count = len(gloves_polygons)

    if gloves_polygons_count == 0:
        current_person.gloves_on = None
    elif gloves_polygons_count == 1:
        current_person.gloves_on = gloves_polygons[0].status
    else:
        max_precition_index = get_polygon_index_with_max_precision(
            gloves_polygons)
        current_person.gloves_on = gloves_polygons[max_precition_index].status

    goggles_polygons = []

    if len(cls_boxes[int(person_class_index.in_goggles)]) > 0:
        goggles_polygons.append(gloves_goggles_polygon(
            cls_boxes[int(person_class_index.in_goggles)][0], True))

    if len(cls_boxes[int(person_class_index.without_goggles)]) > 0:
        goggles_polygons.append(gloves_goggles_polygon(
            cls_boxes[int(person_class_index.without_goggles)][0], False))

    goggles_polygons_count = len(goggles_polygons)

    if goggles_polygons_count == 0:
        current_person.goggles_on = None
    elif goggles_polygons_count == 1:
        current_person.goggles_on = goggles_polygons[0].status
    else:
        max_precition_index = get_polygon_index_with_max_precision(
            goggles_polygons)
        current_person.goggles_on = goggles_polygons[max_precition_index].status

    return current_person


def filter_cls_boxes(cls_boxes):
    for i, boxes in enumerate(cls_boxes):
        boxes_new = []

        for _, box in enumerate(boxes):
            if box[4] >= Treshold:
                boxes_new.append(box)

        cls_boxes[i] = boxes_new

    return cls_boxes


def detect_violations(current_person):
    violations = []

    gloves_violation_message = "Work without gloves"
    goggles_violation_message = "Work without goggles"
    without_hardhat_message = "Work without hardhat"

    if current_person.gloves_on == False:

        if len(person_history) == 0:
            violations.append(gloves_violation_message)

        for i, person in reversed(list(enumerate(person_history))):
            if person.gloves_on is None:
                if i == 0:
                    violations.append(gloves_violation_message)
                    break
                else:
                    continue

            if person.gloves_on == True:
                violations.append(gloves_violation_message)
                break
            else:
                break

    if current_person.goggles_on == False:

        if len(person_history) == 0:
            violations.append(goggles_violation_message)

        for i, person in reversed(list(enumerate(person_history))):

            if person.goggles_on is None:
                if i == 0:
                    violations.append(goggles_violation_message)
                    break
                continue

            if person.goggles_on == True:
                violations.append(goggles_violation_message)
                break
            else:
                break

    if current_person.status == person_hardhat_status.Without_Hardhat:
        if len(person_history) == 0:
            violations.append(without_hardhat_message)

        reversed_person_history = person_history[::-1]

        for i, person in enumerate(reversed_person_history):

            if person.status == person_hardhat_status.Undefined:
                if i == 0:
                    violations.append(without_hardhat_message)
                    break
                continue

            if person.status == person_hardhat_status.In_Hardhat:
                violations.append(without_hardhat_message)
                break
            if person.status == person_hardhat_status.In_Hood:
                for j in reversed(range(0, i - 1)):
                    # pprint(reversed_person_history[j][0])
                    if reversed_person_history[j].status is person_hardhat_status.Undefined:
                        if j == 0:
                            violations.append(without_hardhat_message)
                            break
                        continue

                    if reversed_person_history[j].status is person_hardhat_status.In_Hood:
                        if j == 0:
                            violations.append(without_hardhat_message)
                            break
                        continue

                    if reversed_person_history[j].status is person_hardhat_status.In_Hardhat:
                        violations.append(without_hardhat_message)
                        break

                    if reversed_person_history[j].status is person_hardhat_status.Without_Hardhat:
                        break
            if person.status == person_hardhat_status.Without_Hardhat:
                break

    return violations


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

    im_list = list(im_list)
    im_list.sort()

    # Define the codec and create VideoWriter object
    xvid_codec = 1145656920
    frame_width = 1280
    frame_heigth = 960
    video = cv2.VideoWriter(args.output_video, xvid_codec,
                            1.0, (frame_width, frame_heigth))

    csv = open(args.csv_path, 'w+')

    for _, im_name in enumerate(im_list):
        # out_name = os.path.join(
        #     args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf'))

        im = cv2.imread(im_name)

        logger.info('Processing {}'.format(
            color.RED + color.BOLD + im_name + color.END + color.END))

        with c2_utils.NamedCudaScope(0):
            cls_boxes, _, _ = infer_engine.im_detect_all(
                model, im, None)

            current_person = convert_to_person(filter_cls_boxes(cls_boxes))
            violations = detect_violations(current_person)
            person_history.append(current_person)

        img_bbox = vis_utils.vis_one_image_opencv(
            im,
            cls_boxes,
            None,
            None,
            thresh=Treshold,
            show_box=True,
            dataset=dummy_coco_dataset,
            show_class=True)

        frame = cv2.resize(img_bbox, (frame_width, frame_heigth),
                           interpolation=cv2.INTER_CUBIC)
        video.write(frame)

        if len(violations) > 0:
            file_name = os.path.split(im_name)[1]
            file_without_ext = os.path.splitext(file_name)[0]

            for _, text in enumerate(violations):
                csv.write(text + "," + str(file_without_ext) + '\n')

    video.release()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
