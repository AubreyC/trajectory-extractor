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
import scipy.optimize as opt

from traj_ext.utils.mathutil import *
from traj_ext.box3D_fitting import Box3D_utils

from traj_ext.tracker import cameramodel
from traj_ext.utils.mathutil import *

from traj_ext.object_det import det_object
from traj_ext.box3D_fitting import box3D_object

# Root directory of the project
CAMERA_CFG_1_PATH = 'dataset/test/varna_20190125_153327_0_900/varna_area1_camera_street_cfg.yml';
IMG_PATH = 'dataset/test/varna_20190125_153327_0_900/img/varna_20190125_153327_0_900_0000000000.jpg';
DET_PATH = 'dataset/test/varna_20190125_153327_0_900/output/det_crop/csv/varna_20190125_153327_0_900_0000000000_det.csv';

if __name__ == '__main__':

    ##########################################################
    # Input to the script:
    ##########################################################

    # Print instructions
    print("############################################################\n")
    print("Test optim 3D box mono with manual control")

    print("Instruction:")
    print("    - w,a,s,d moves box along X,Y axis")
    print("    - z,x rotates the box along Z axis")
    print("    - r,f changes width of the box")
    print("    - r,f changes length of the box")
    print("    - t,g changes width of the box")
    print("    - y,h changes height of the box")
    print("    - n next detection")
    print("    - Press q to quit\n")

    parser = argparse.ArgumentParser(description='Test optim 3D box mono with manual control');
    parser.add_argument('-camera', dest="cam_model_path", type=str, help='Path of the camera model yml', default=CAMERA_CFG_1_PATH);
    parser.add_argument('-image', dest="image_path", type=str, help='Path of the image', default = IMG_PATH);
    parser.add_argument('-det', dest="det_path", type=str, help='Path of the detections files', default = DET_PATH);

    args = parser.parse_args();

    ##########################################################
    # Camera Parameters
    ##########################################################

    cam_model_1 = cameramodel.CameraModel.read_from_yml(args.cam_model_path);

    cam_scale_factor = 1.0;
    cam_model_1.apply_scale_factor(cam_scale_factor,cam_scale_factor);

    image_path = args.image_path;

    # CSV name management
    csv_path = args.det_path;
    det_object_list = det_object.DetObject.from_csv(csv_path, expand_mask = True);

    # Nober of detections
    nb_det = len(det_object_list);

    # Open Image
    im_1 = cv2.imread(image_path);
    im_1 = cv2.resize(im_1,None,fx=cam_scale_factor, fy=cam_scale_factor, interpolation = cv2.INTER_CUBIC)

    exit_flag = False;
    for det in det_object_list:

        if not exit_flag:
            if det.label == 'car':

                det_scaled = det.to_scale(cam_scale_factor, cam_scale_factor);

                im_size_1 = (im_1.shape[0], im_1.shape[1]);

                l = float(5);
                w = float(2);
                h = float(-1.6);
                box_size_lwh = [l,w,h];

                # Start the optimization
                box3D_result = Box3D_utils.find_3Dbox(det_scaled.det_mask, det_scaled.det_2Dbox, cam_model_1, im_size_1, box_size_lwh);

                print('Percent overlap: {}'.format(box3D_result.percent_overlap))

                while True:

                    im_current_1 = copy.copy(im_1);

                    box3D_result.display_on_image(im_current_1, cam_model_1);

                    # Mask box
                    mask_box_1 = box3D_result.create_mask(cam_model_1, im_size_1);

                    o_1, mo_1, mo_1_b = Box3D_utils.overlap_mask(det_scaled.det_mask, mask_box_1);
                    print("Overlap total: {}".format(o_1));

                    # Draw the mask: Intersection \ Union and Instance mask (from mask-RCNN)
                    im_current_1 = det_object.draw_mask(im_current_1, mask_box_1, (0,0,255));
                    im_current_1 = det_object.draw_mask(im_current_1, det_scaled.det_mask, (0,255,255));

                    # Show the image
                    cv2.imshow("Camera 1", im_current_1)

                    key = cv2.waitKey(0) & 0xFF

                    # if the 'c' key is pressed, reset the cropping region
                    if key == ord("q"):
                        exit_flag = True;
                        break;

                    elif key == ord("w"):
                        box3D_result.x = box3D_result.x + 0.1;
                    elif key == ord("s"):
                        box3D_result.x = box3D_result.x - 0.1;

                    elif key == ord("d"):
                        box3D_result.y = box3D_result.y + 0.1;
                    elif key == ord("a"):
                        box3D_result.y = box3D_result.y - 0.1;

                    elif key == ord("x"):
                        box3D_result.psi_rad = box3D_result.psi_rad + 0.1;
                    elif key == ord("z"):
                        box3D_result.psi_rad = box3D_result.psi_rad - 0.1;

                    elif key == ord("r"):
                        box3D_result.length = box3D_result.length + 0.1;
                    elif key == ord("f"):
                        box3D_result.length = box3D_result.length - 0.1;

                    elif key == ord("t"):
                        box3D_result.width = box3D_result.width + 0.1;
                    elif key == ord("g"):
                        box3D_result.width = box3D_result.width - 0.1;

                    elif key == ord("y"):
                        box3D_result.height = box3D_result.height + 0.1;
                    elif key == ord("h"):
                        box3D_result.height = box3D_result.height - 0.1;

                    elif key == ord("n"):
                        break;

    cv2.destroyAllWindows()



