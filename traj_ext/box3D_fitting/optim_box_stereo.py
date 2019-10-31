# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-05-17 19:52:57
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

import numpy as np
import time
import cv2
import copy
from scipy.optimize import linear_sum_assignment
import os
import sys
import argparse
from project_rectangle import *
import scipy.optimize as opt

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../'));

from traj_ext.tracker import cameramodel as cm
from traj_ext.utils.mathutil import *
from traj_ext.box3D_fitting import Box3D_utils
from traj_ext.camera_calib import calib_utils
from traj_ext.object_det.mask_rcnn import detect_utils

CAMERA_CFG_1_PATH = '../camera_calib/calib_file/auburn_camera_street_1_cfg.yml';
CAMERA_CFG_2_PATH = '../camera_calib/calib_file/auburn_camera_street_2_cfg_2.yml';


IMG_1_PATH = os.path.join(ROOT_DIR,'box3D_fitting/data_test/auburn1_20171005_183440_158258.jpg');
DET_1_PATH = os.path.join(ROOT_DIR,'box3D_fitting/data_test/auburn1_20171005_183440_158258.csv');

IMG_2_PATH = os.path.join(ROOT_DIR,'box3D_fitting/data_test/auburn2_20171005_183520_121353.jpg');
DET_2_PATH = os.path.join(ROOT_DIR,'box3D_fitting/data_test/auburn2_20171005_183520_121353.csv');

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

def draw_boundingbox(im, r):

    for i in range(0, len(r['rois'])):
        x_1 = int(r['rois'][i][1]);
        y_1 = int(r['rois'][i][0]);
        x_2 = int(r['rois'][i][3]);
        y_2 = int(r['rois'][i][2]);

        tl = (x_1, y_1)
        br = (x_2, y_2)
        # label = result['label']
        # confidence = result['confidence']
        text = 'Mask: {}'.format(i)
        im = cv2.rectangle(im, tl, br, (255, 0, 0), 1)
        im = cv2.putText(im, text, tl, cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)


        # if True:
        #     clrImg = np.zeros(im.shape, im.dtype)
        #     clrImg[:,:] = (255, 255, 0)

        #     m = np.array(r['masks'][:,:,i], dtype = "uint8")
        #     clrMask = cv2.bitwise_and(clrImg, clrImg, mask=m);

        #     cv2.addWeighted(im, 1.0, clrMask, 0.5, 0.0, im)


    return im;


def func_cons_1(opti_params):
    phi = opti_params[0];
    # phi = np.rad2deg(phi)
    x = opti_params[1];
    y = opti_params[2];
    z = opti_params[3];
    l = opti_params[4];
    w = opti_params[5];
    h = opti_params[6];

    return l-3.5;


def func_cons_2(opti_params):
    phi = opti_params[0];
    # phi = np.rad2deg(phi)
    x = opti_params[1];
    y = opti_params[2];
    z = opti_params[3];
    l = opti_params[4];
    w = opti_params[5];
    h = opti_params[6];

    return 5.5 - l;

def func_cons_3(opti_params):
    phi = opti_params[0];
    # phi = np.rad2deg(phi)
    x = opti_params[1];
    y = opti_params[2];
    z = opti_params[3];
    l = opti_params[4];
    w = opti_params[5];
    h = opti_params[6];

    return w-1.5;


def func_cons_4(opti_params):
    phi = opti_params[0];
    # phi = np.rad2deg(phi)
    x = opti_params[1];
    y = opti_params[2];
    z = opti_params[3];
    l = opti_params[4];
    w = opti_params[5];
    h = opti_params[6];

    return 3 - w;



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

    im_1 = cv2.imread(IMG_1_PATH);
    im_2 = cv2.imread(IMG_2_PATH);

    r_1 = detect_utils.read_detection_csv(DET_1_PATH, class_names);
    r_2 = detect_utils.read_detection_csv(DET_2_PATH, class_names);

    # Detection mask
    ind_det_1 = 0;
    ind_det_2 = 10;

    # Compute rough position
    pt_image_x = (r_1['rois'][ind_det_1, 1] + r_1['rois'][ind_det_1, 3])/2
    pt_image_y = (r_1['rois'][ind_det_1, 0] + r_1['rois'][ind_det_1, 2])/2

    pt_image = (int(pt_image_x), int(pt_image_y));
    print(pt_image)
    pos_FNED_init = cam_model_1.projection_ground(0, pt_image);
    pos_FNED_init.shape = (1,3);
    print("Rough position: {}".format(str(pos_FNED_init)));

    mask_1 = r_1['masks'][:,:,ind_det_1];
    mask_2 = r_2['masks'][:,:,ind_det_2];

    mask_1 = mask_1.astype(np.bool);
    mask_2 = mask_2.astype(np.bool);

    im_size_1 = (im_1.shape[0], im_1.shape[1]);
    im_size_2 = (im_2.shape[0], im_2.shape[1]);

    # Construct points
    phi = 10;
    x = pos_FNED_init[0,0];
    y = pos_FNED_init[0,1];
    z = 0;
    l = 5;
    w = 2;
    h = -1.6;

    # x = -7.7;
    # y = 51.0;
    # z = 0;
    # l = 4.5;
    # w = 1.8;
    # h = -1.6;
    # phi = 51;

    # param = opt.minimize(func, p_init,method='Nelder-Mead', args=(im_size_1, im_size_2, cam_model_1, cam_model_2, mask_1, mask_2, param_fix), options={'xtol': 1e-8, 'disp': True});
    # cons = ({'type': 'ineq', 'fun': lambda x : func_cons_1(x)},\
    #        {'type': 'ineq', 'fun': lambda x : func_cons_2(x)}, \
    #        {'type': 'ineq', 'fun': lambda x : func_cons_3(x)}, \
    #        {'type': 'ineq', 'fun': lambda x : func_cons_4(x)});


    param_min = None;
    for phi in range(0,180,60):

        param_box = [phi, x, y, z, l, w, h];


        param_opt = [phi, x, y];
        param_fix = [z, l, w, h];
        p_init = param_opt;

        # Good method: COBYLA, Powell
        param = opt.minimize(Box3D_utils.compute_cost_stero, p_init, method='Powell', args=(im_size_1, im_size_2, cam_model_1, cam_model_2, mask_1, mask_2, param_fix), options={'maxfev': 50, 'disp': True});
        if param_min is None:
            param_min = param;

        if param.fun < param_min.fun:
            param_min = param;

    param = param_min;

    # param = opt.fmin_slsqp(func, p_init, args=(im_size_1, im_size_2, cam_model_1, cam_model_2, mask_1, mask_2, param_fix), disp=True);

    #Define Constraint: Only on time


    # bounds = [(0,360), (x-5, x+5), (y-5, y+5)]
    # print bounds
    # # bounds = [(0,360), (-30, 30), (0, 60), (0, 0), (2, 5), (1, 3), (-1, -2)]
    # x = opt.brute(func, bounds, Ns=10, args=(im_size_1, im_size_2, cam_model_1, cam_model_2, mask_1, mask_2, param_fix), disp=True);
    # print(x)
    # phi = x[0];
    # x = x[1];
    # y = x[2];

    # param = opt.differential_evolution(func, bounds,  args=(im_size_1, im_size_2, cam_model_1, cam_model_2, mask_1, mask_2, param_fix), disp=True)

    print('With param [phi, x, y]: %s' %(str(param.x)));

    # phi = param.x[0];
    # x = param.x[1];
    # y = param.x[2];
    # z = param.x[3];
    # l = param.x[4];
    # w = param.x[5];
    # h = param.x[6];

    phi = param.x[0];
    x = param.x[1];
    y = param.x[2];

    z = param_fix[0];
    l = param_fix[1];
    w = param_fix[2];
    h = param_fix[3];



    # keep looping until the 'q' key is pressed
    while True:

        im_current_1 = copy.copy(im_1);
        im_current_2 = copy.copy(im_2);

        param_box = [phi, x, y, z, l, w, h];

        list_pt_F = Box3D_utils.create_3Dbox(param_box);

        im_current_1 = Box3D_utils.draw_3Dbox(im_current_1, cam_model_1, list_pt_F);
        im_current_2 = Box3D_utils.draw_3Dbox(im_current_2, cam_model_2, list_pt_F);

        # Mask box
        mask_box_1 = Box3D_utils.create_mask(im_size_1, cam_model_1, list_pt_F);
        mask_box_2 = Box3D_utils.create_mask(im_size_2, cam_model_2, list_pt_F);

        o_1, mo_1, mo_1_b = Box3D_utils.overlap_mask(mask_1, mask_box_1);
        o_2, mo_2, mo_2_b = Box3D_utils.overlap_mask(mask_2, mask_box_2);

        print("Overlap total: {}".format(o_1))

        im_current_1 = draw_boundingbox(im_current_1, r_1);
        im_current_2 = draw_boundingbox(im_current_2, r_2);

        # im_current_1 = draw_mask(im_current_1, mask_1, (255,255,0));
        # im_current_2 = draw_mask(im_current_2, mask_2, (255,255,0));

        im_current_1 = Box3D_utils.draw_mask(im_current_1, mo_1, (0,0,255));
        im_current_1 = Box3D_utils.draw_mask(im_current_1, mo_1_b, (0,255,255));


        im_current_2 = Box3D_utils.draw_mask(im_current_2, mo_2, (0,0,255));
        im_current_2 = Box3D_utils.draw_mask(im_current_2, mo_2_b, (0,255,255));


        cv2.imshow("Camera 1", im_current_1)
        cv2.imshow("Camera 2", im_current_2)

        print("Box Params:");
        print("x = {};".format(x));
        print("y = {};".format(y));
        print("z = {};".format(z));
        print("l = {};".format(l));
        print("w = {};".format(w));
        print("h = {};".format(h));
        print("phi = {};".format(phi));
        print("");

        key = cv2.waitKey(0) & 0xFF
        # if the 'c' key is pressed, reset the cropping region
        if key == ord("q"):
            break
        elif key == ord("w"):
            x = x + 0.1;
        elif key == ord("s"):
            x = x - 0.1;

        elif key == ord("d"):
            y = y + 0.1;
        elif key == ord("a"):
            y = y - 0.1;

        elif key == ord("x"):
            phi = phi + 1;
        elif key == ord("z"):
            phi = phi - 1;

        elif key == ord("r"):
            l = l + 0.1;
        elif key == ord("f"):
            l = l - 0.1;

        elif key == ord("t"):
            w = w + 0.1;
        elif key == ord("g"):
            w = w - 0.1;

        elif key == ord("y"):
            h = h + 0.1;
        elif key == ord("h"):
            h = h - 0.1;

    cv2.destroyAllWindows()



