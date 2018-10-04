#!/usr/bin/env python2

"""Perform inference on one or more datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import pprint
import sys
import time
import numpy as np

import caffe2
from caffe2.python import workspace, core

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.utils.io import cache_url
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
import detectron.core.test_engine as infer_engine
import detectron.core.test as test
import detectron.utils.c2 as c2_utils
import detectron.utils.model_convert_utils as mutils

from detectron.custom.color import color

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        required=True,
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        required=True,
        type=str)
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str)
    parser.add_argument(
        'im_or_folder',
        help='image or folder of images',
        default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


net_file = '/home/user/vilin/detectron-output/hardhat_gloves_goggles_5/train/hardhat_gloves_goggles_3_train/generalized_rcnn/net.pbtxt'


def to_console_output_string(source_string):
    return '\n  ' + color.RED + color.BOLD + source_string + color.END + color.END


def initialize_model():
    merge_cfg_from_file(args.cfg_file)
    cfg.NUM_GPUS = 1

    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg()

    model = infer_engine.initialize_model_from_cfg(args.weights)

    return model


def main(args):
    logging.disable(logging.WARNING)

    model = initialize_model()

    # print(model.net.blobs)

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for img in im_list:
        with c2_utils.NamedCudaScope(0):
            im = cv2.imread(img)
            test.im_detect_all(model, im, None)

        print('img max pixel value = ' + str(np.amax(im)))
        # result = convert_pkl_to_pb.run_model_cfg(args, im, 'conv5')
        # print('result = {}'.format(result))

    blobs = mutils.get_ws_blobs()

    # f = open("/tmp/blobs_keys.txt", "w+")
    # for key in blobs.keys():
    #     f.write('{}                                                           # {} \r\n'.format(
    #         key, blobs[key].shape))
    # f.close()

    cv2.imwrite('/tmp/test.jpg',
                np.multiply(blobs['gpu_0/res2_2_sum'][0][0], 100))

    print(to_console_output_string('_concat_roi_feat') +
          ' = {} \n'.format(blobs['gpu_0/_concat_roi_feat']))

    print(to_console_output_string('res5_2_sum [0][0]') +
          ' = {} \n'.format(blobs['gpu_0/res5_2_sum'][0][0]))  # (2048, 38, 25)

    # print(to_console_output_string('res5_2_sum[0][0]') +
    #       ' = {} \n'.format(blobs['gpu_0/res5_2_sum'][0][1999]))

    print(to_console_output_string('roi_feat_shuffled') +
          # (1000, 256, 7, 7)
          ' = {} \n'.format(np.amax(blobs['gpu_0/roi_feat_shuffled'])))

    print(to_console_output_string('roi_feat') +
          ' = {} \n'.format(blobs['gpu_0/roi_feat'][0][0]))  # (1000, 256, 7, 7)

    print(to_console_output_string('bbox_pred.shape') +
          ' = {} \n'.format(blobs['gpu_0/bbox_pred'].shape))  # (1000, 32)

    print(to_console_output_string(
        'bbox_pred[0]') + ' = {} \n'.format(blobs['gpu_0/bbox_pred'][0]))

    print(' rois[0].shape = {} \n'.format(
        blobs['gpu_0/rois'].shape))  # (1000, 5)

    print(' rois[0] = {} \n'.format(blobs['gpu_0/rois'][0]))

    print(' rpn_rois_fpn6.shape = {} \n'.format(
        blobs['gpu_0/rpn_rois_fpn6'].shape))

    print(' rpn_rois_fpn5[0] = {} \n'.format(blobs['gpu_0/rpn_rois_fpn5'][0]))

    print(' cls_prob.shape = {} \n'.format(blobs['gpu_0/cls_prob'].shape))


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = setup_logging(__name__)

    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    main(args)
