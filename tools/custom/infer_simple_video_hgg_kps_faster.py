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
from sympy import *
from detectron.core.test import im_detect_keypoints, keypoint_results
import numpy as np

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


CHANCE_THRESHOLD = 0.7
left_ankle_class_index = 15
right_ankle_class_index = 16

danger_zone_1600_1200 = [136, 453, 587, 894]
danger_zone_1280_960 = [78, 388, 377, 748]
# x1, y1, x2, y2
danger_zone = danger_zone_1280_960

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
    def __init__(self, bbox, status, gloves_on, goggles_on, in_danger_zone):
        self.bbox = bbox
        self.status = status
        self.gloves_on = gloves_on
        self.goggles_on = goggles_on
        self.in_danger_zone = in_danger_zone


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


def get_polygon_index_with_max_precision(bboxes):
    precitions = []

    for _, polygon in enumerate(bboxes):
        precitions.append(polygon.bbox[4])

    return int(max(precitions))


def convert_to_person(cls_boxes, is_in_danger_zone):
    current_person = person(None, None, None, None, is_in_danger_zone)

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
    danger_zone_message = "Work in danger zone"

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

    if current_person.in_danger_zone:
        if len(person_history) == 0:
            violations.append(danger_zone_message)

        for i, person in reversed(list(enumerate(person_history))):
            if person.in_danger_zone:
                break

            if not person.in_danger_zone:
                violations.append(danger_zone_message)
                break

    return violations


keypoints_model_path = '/home/user/vilin/detectron-input/pretrained_models/model_final.pkl'
keypoints_config_file_path = '/home/user/vilin/detectron-input/pretrained_models/e2e_mask_rcnn_R-50-FPN_1x.yaml'


def main(args):
    # logging.disable("warning")
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False, make_immutable=False)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    im_list = list(im_list)
    im_list.sort()

    # Define the codec and create VideoWriter object
    xvid_codec = 1145656920
    frame_width = 1600
    frame_heigth = 1200
    video = cv2.VideoWriter(args.output_video, xvid_codec,
                            1.0, (frame_width, frame_heigth))

    csv = open(args.csv_path, 'w+')

    for _, im_name in enumerate(im_list):
        # out_name = os.path.join(
        #     args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf'))

        im = cv2.imread(im_name)

        logger.error('Processing {}'.format(
            color.RED + color.BOLD + im_name + color.END + color.END))

        merge_cfg_from_file(args.cfg)
        args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
        hardhat_model = infer_engine.initialize_model_from_cfg(
            args.weights)

        with c2_utils.NamedCudaScope(0):
            cls_boxes, _, _, im_scale = infer_engine.im_detect_all(
                hardhat_model, im, None)

        merge_cfg_from_file(keypoints_config_file_path)
        keypoints_weights = cache_url(keypoints_model_path, cfg.DOWNLOAD_CACHE)
        keypoints_model = infer_engine.initialize_model_from_cfg(
            keypoints_weights)
        assert_and_infer_cfg(cache_urls=False, make_immutable=False)

        if cls_boxes[1].shape[0] > 0:

            with c2_utils.NamedCudaScope(0):
                cls_boxes_person_coord = []
                for _, box in enumerate(cls_boxes[1]):
                    cls_boxes_person_coord.append(box[:-1])
                cls_boxes_person_coord = np.asarray(cls_boxes_person_coord)

                heatmaps = im_detect_keypoints(
                    keypoints_model, im_scale, cls_boxes_person_coord)

                # print(color.BLUE + color.BOLD +
                #       str(heatmaps) + color.END + color.END)

                cls_keyps = keypoint_results(
                    cls_boxes, heatmaps, cls_boxes_person_coord)
        else:
            cls_keyps = None

        # with c2_utils.NamedCudaScope(0):
        #     _, _, cls_keyps = infer_engine.im_detect_all(
        #         keypoints_model, im, None)

        # persons = []
        # person_kps = []

        # for p in range(len(cls_boxes[int(person_class_index.in_hardhat)])):
        #     person = cls_boxes[int(person_class_index.in_hardhat)][p]
        #     person_key = cls_keyps[int(person_class_index.in_hardhat)][p]

        #     if person[4] > Treshold:
        #         persons.append(person)
        #         person_kps.append(person_key)

        # cls_keyps[int(person_class_index.in_hardhat)] = person_kps

        # ankles = collect_ankles(cls_keyps[int(person_class_index.in_hardhat)])
        # is_in_danger_zone = check_ankles_in_danger_zone(ankles, danger_zone)
        is_in_danger_zone = False

        current_person = convert_to_person(
            filter_cls_boxes(cls_boxes), is_in_danger_zone)
        violations = detect_violations(current_person)
        person_history.append(current_person)

        # print("cls_boxes[without goggles] = {}".format(
        #     cls_boxes[]))

        # pprint("hardhat = {}".format(current_person.status))
        # pprint("goggles on = {}".format(current_person.goggles_on))
        # pprint("gloves on = {}".format(current_person.gloves_on))

        # print(color.BLUE + color.BOLD + str(cls_keyps) + color.END + color.END)
        # print(color.BLUE + color.BOLD +
        #       str(len(cls_keyps)) + color.END + color.END)

        # cls_keyps.append(cls_keyps[1])
        # cls_keyps.append(cls_keyps[1])
        # cls_keyps.append(cls_keyps[1])
        # cls_keyps.append(cls_keyps[1])

        # print(color.BLUE + color.BOLD +
        #       str(len(cls_keyps)) + color.END + color.END)

        img_with_bboxes = vis_utils.vis_one_image_opencv(
            im,
            cls_boxes,
            segms=None,
            keypoints=cls_keyps,
            thresh=Treshold,
            show_box=True,
            dataset=dummy_coco_dataset,
            show_class=True)

        frame = cv2.resize(img_with_bboxes, (frame_width, frame_heigth),
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
