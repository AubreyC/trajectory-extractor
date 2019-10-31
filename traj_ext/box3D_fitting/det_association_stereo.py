# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-05-17 19:52:57
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

# The idea here is:
# - Based on detection on Image 1: compute a 3D region of where the detected object can be
# - Re-project this 3D region (3D polygon) on image 2
# - Find detection on image 2 that intersect with the projected potential region (from step 2)
# Not sure this is the most efficient way to do it

import numpy as np
import time
import cv2
import copy
from scipy.optimize import linear_sum_assignment
import os
import sys
import argparse
import scipy.optimize as opt
from multiprocessing.dummy import Pool as ThreadPool

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../'));

from traj_ext.box3D_fitting import Box3D_utils

from traj_ext.object_det.mask_rcnn import detect_utils
from traj_ext.tracker import cameramodel as cm
from traj_ext.camera_calib import calib_utils

from traj_ext.tracker import cameramodel as cm
from traj_ext.utils.mathutil import *
from traj_ext.object_det.mask_rcnn import detect_utils


CAMERA_CFG_1_PATH = os.path.join(ROOT_DIR,'camera_calib/calib_file/auburn_camera_street_1_cfg.yml');
CAMERA_CFG_2_PATH = os.path.join(ROOT_DIR,'camera_calib/calib_file/auburn_camera_street_2_cfg_2.yml');

IMAGE_1_DIR = os.path.join(ROOT_DIR, "box3D_fitting/data_test")
IMAGE_1_DIR = os.path.abspath(IMAGE_1_DIR);

DET_1_DIR = os.path.join(ROOT_DIR, "box3D_fitting/data_test")
DET_1_DIR = os.path.abspath(DET_1_DIR);

IMAGE_2_DIR = os.path.join(ROOT_DIR, "box3D_fitting/data_test")
IMAGE_2_DIR = os.path.abspath(IMAGE_1_DIR);

DET_2_DIR = os.path.join(ROOT_DIR, "box3D_fitting/data_test")
DET_2_DIR = os.path.abspath(DET_1_DIR);

OUTPUT_1_IMG = os.path.join(ROOT_DIR, "output/auburn1_20171005_183440/output/box_3/")

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def find_boxes(data_det,cam_model, im_size):


    box_car = [5,2,-1.6];
    box_person = [0.8, 0.8, -1.9];
    box_bus = [13, 3.5, -2.5];


    nb_det = len(data_det);
    array_r = [];
    for det_ind in range(0, nb_det):

        det_dict = data_det[det_ind];

        if det_dict['label'] == 'car' :

            r ={};

            r['mask'] = det_dict['mask'];
            r['roi'] = det_dict['roi'];
            r['det_id'] = det_dict['det_id'];

            r['cam_model'] = cam_model;
            r['im_size'] =  im_size;
            r['box_size'] = box_car;

            array_r.append(r);

        if det_dict['label'] == 'person' :

            r ={};

            r['mask'] = det_dict['mask'];
            r['roi'] = det_dict['roi'];
            r['det_id'] = det_dict['det_id'];

            r['cam_model'] = cam_model;
            r['im_size'] =  im_size;
            r['box_size'] = box_person;

            array_r.append(r);

        if det_dict['label'] == 'bus' :

            r ={};

            r['mask'] = det_dict['mask'];
            r['roi'] = det_dict['roi'];
            r['det_id'] = det_dict['det_id'];

            r['cam_model'] = cam_model;
            r['im_size'] =  im_size;
            r['box_size'] = box_bus;

            array_r.append(r);

    pool = ThreadPool(10);
    results = pool.map(Box3D_utils.find_3Dbox_multithread, array_r);

    return results;

if __name__ == '__main__':

    ##########################################################
    # Input to the script:
    ##########################################################


    parser = argparse.ArgumentParser(description='Test ');
    parser.add_argument('--camera_1_cfg',dest="camera_1_cfg", default=CAMERA_CFG_1_PATH, type=str, help='Camera 1 yml config file');
    parser.add_argument('--camera_2_cfg',dest="camera_2_cfg", default=CAMERA_CFG_2_PATH, type=str, help='Camera 2 yml config file');

    # parser.add_argument('--camera_sat_cfg',dest="camera_sat_cfg", default=CAMERA_CFG_SAT_PATH, type=str, help='Camera Satellite yml config file');
    args = parser.parse_args();


    # Construct camera model
    cam_model_1 = calib_utils.read_camera_calibration(args.camera_1_cfg);
    cam_model_2 = calib_utils.read_camera_calibration(args.camera_2_cfg);

    # Load images
    list_file_1 = os.listdir(IMAGE_1_DIR);
    list_file_1.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # Load images
    list_file_2 = os.listdir(IMAGE_2_DIR);
    list_file_2.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    image_name_1 = 'auburn1_20171005_183440_0000050230.jpg';
    image_name_2 = 'auburn2_20171005_183520_0000013298.jpg';

    if not image_name_1 in list_file_1:
        print("Error: Image {} not in {}".format(image_name_1, IMAGE_1_DIR));
        sys.exit();

    if not image_name_2 in list_file_2:
        print("Error: Image {} not in {}".format(image_name_2, IMAGE_2_DIR));
        sys.exit();

    im_1 = cv2.imread(os.path.join(IMAGE_1_DIR, image_name_1));
    im_2 = cv2.imread(os.path.join(IMAGE_2_DIR, image_name_2));
    im_size_1 = (im_1.shape[0], im_1.shape[1]);
    im_size_2 = (im_2.shape[0], im_2.shape[1]);

    im_current_1 = copy.copy(im_1);
    im_current_2 = copy.copy(im_2);

    # CSV name management
    csv_name_1 = image_name_1.split('.')[0] + '_det.csv';
    # r_1 = detect_utils.read_detection_csv(os.path.join(DET_1_DIR, csv_name_1), class_names);
    try:
        data_det_1 = detect_utils.read_dict_csv(os.path.join(DET_1_DIR, csv_name_1));
    except Exception as e:
        print(e);
        print("Could not open detection csv {}".format(os.path.join(DET_1_DIR, csv_name_1)));
        sys.exit();

    csv_name_2 = image_name_2.split('.')[0] + '_det.csv';
    # r_2 = detect_utils.read_detection_csv(os.path.join(DET_2_DIR, csv_name_2), class_names);
    try:
        data_det_2 = detect_utils.read_dict_csv(os.path.join(DET_2_DIR, csv_name_2));
    except Exception as e:
        print(e);
        print("Could not open detection csv {}".format(os.path.join(DET_2_DIR, csv_name_2)));
        sys.exit();

    result_box_1 = find_boxes(data_det_1, cam_model_1, im_size_1);
    result_box_2 = find_boxes(data_det_2, cam_model_2, im_size_2);

    for result in result_box_1:

        param_box = result['box_3D'];
        mask_1 = result['mask'];

        list_pt_F = Box3D_utils.create_3Dbox(param_box);

        im_current_1 = Box3D_utils.draw_3Dbox(im_current_1, cam_model_1, list_pt_F);

        # Mask box
        mask_box_1 = Box3D_utils.create_mask(im_size_1, cam_model_1, list_pt_F);

        o_1, mo_1, mo_1_b = Box3D_utils.overlap_mask(mask_1, mask_box_1);

        print("Overlap total: {}".format(o_1));


        im_current_1 = Box3D_utils.draw_mask(im_current_1, mask_box_1, (0,0,255));
        im_current_1 = Box3D_utils.draw_mask(im_current_1, mask_1, (0,255,255));

        # im_current_1 = draw_boundingbox(im_current_1, r_1);

    for result in result_box_2:

        param_box = result['box_3D'];
        mask_1 = result['mask'];

        list_pt_F = Box3D_utils.create_3Dbox(param_box);

        im_current_2 = Box3D_utils.draw_3Dbox(im_current_2, cam_model_2, list_pt_F);

        # Mask box
        mask_box_1 = Box3D_utils.create_mask(im_size_2, cam_model_2, list_pt_F);

        o_1, mo_1, mo_1_b = Box3D_utils.overlap_mask(mask_1, mask_box_1);

        # print("Overlap total: {}".format(o_1))


        im_current_2 = Box3D_utils.draw_mask(im_current_2, mask_box_1, (0,0,255));
        im_current_2 = Box3D_utils.draw_mask(im_current_2, mask_1, (0,255,255));

        # im_current_2 = draw_boundingbox(im_current_2, r_2);

    # Construct association matrix:
    cost = np.zeros((len(result_box_1), len(result_box_2)));
    print(cost)
    ind_1 = 0;
    ind_2 = 0;
    for ind_1 in np.arange(len(result_box_1)):
        result_1 = result_box_1[ind_1];

        for ind_2 in np.arange(len(result_box_2)):
            result_2 = result_box_2[ind_2];

            param_box_1 = result_1['box_3D'];
            param_box_2 = result_2['box_3D'];

            pos_1 = np.asarray(param_box_1[1:4]);
            pos_2 = np.asarray(param_box_2[1:4]);
            dist = np.linalg.norm(pos_1 - pos_2);


            cost[ind_1, ind_2] = dist;

            if dist < 5:
                print("Match Image 1: {} with Image 2: {}".format(result_1['det_id'], result_2['det_id']))
    print(cost)

    cv2.imshow("im_current_1", im_current_1)
    cv2.imshow("im_current_2", im_current_2)

    cv2.waitKey(0)


    #         cv2.waitKey(0)
    # cv2.destroyAllWindows()
