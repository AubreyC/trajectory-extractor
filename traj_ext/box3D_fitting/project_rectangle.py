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
import argparse
import sys
import os

from traj_ext.tracker import cameramodel
from traj_ext.utils.mathutil import *

from traj_ext.box3D_fitting import box3D_object


CAMERA_CFG_1_PATH = '../camera_calib/calib_file/biloxi/biloxi_cam_cfg.yml';
CAMERA_CFG_2_PATH = '../camera_calib/calib_file/biloxi/biloxi_sat_cfg.yml';

IMG_1_PATH = '../camera_calib/calib_file/biloxi/biloxi_cam.jpg';
IMG_2_PATH = '../camera_calib/calib_file/biloxi/biloxi_sat.png';

if __name__ == '__main__':

    ##########################################################
    # Input to the script:
    ##########################################################

    # Print instructions
    print("############################################################\n")
    print("Project Rectangle:")
    print("Create a 3D box, moves it in the 3D world frame and project it on the 2 camera images.")

    print("Instruction:")
    print("    - w,a,s,d moves box along X,Y axis")
    print("    - z,x rotates the box along Z axis")
    print("    - r,f changes width of the box")
    print("    - r,f changes length of the box")
    print("    - t,g changes width of the box")
    print("    - y,h changes height of the box")
    print("    - Press q to quit\n")

    parser = argparse.ArgumentParser(description='Project Rectangle');
    parser.add_argument('--camera_1_cfg',dest="camera_1_cfg", default=CAMERA_CFG_1_PATH, type=str, help='Camera 1 yml config file');
    parser.add_argument('--camera_2_cfg',dest="camera_2_cfg", default=CAMERA_CFG_2_PATH, type=str, help='Camera 2 yml config file');

    args = parser.parse_args();

    # Construct camera model
    cam_model_1 = cameramodel.CameraModel.read_from_yml(args.camera_1_cfg);
    cam_model_2 = cameramodel.CameraModel.read_from_yml(args.camera_2_cfg);

    im_1 = cv2.imread(IMG_1_PATH);
    im_2 = cv2.imread(IMG_2_PATH);

    # Construct points
    psi_rad = 0.0;
    x = -0.7;
    y = 0.8;
    z = 0;
    l = 4.5;
    w = 1.8;
    h = -1.6;


    # keep looping until the 'q' key is pressed
    while True:

        im_current_1 = copy.copy(im_1);
        im_current_2 = copy.copy(im_2);

        box3D = box3D_object.Box3DObject(psi_rad, x, y, z, l, w, h);

        # pt = np.array([(x,  y, z)])
        # (pt_img, jacobian) = cv2.projectPoints(pt, rot_CF1_F, trans_CF1_F, cam_matrix_1, dist_coeffs_1)
        # pt_img = (int(pt_img[0][0][0]), int(pt_img[0][0][1]));
        # cv2.circle(im_current, pt_img, 5, (0,255, 255), -1)

        box3D.display_on_image(im_current_1, cam_model_1);
        box3D.display_on_image(im_current_2, cam_model_2);

        cv2.imshow("Camera 1", im_current_1)
        cv2.imshow("Camera 2", im_current_2)

        print("Box Params:");
        print("x = {};".format(x));
        print("y = {};".format(y));
        print("z = {};".format(z));
        print("l = {};".format(l));
        print("w = {};".format(w));
        print("h = {};".format(h));
        print("psi = {};".format(psi_rad));
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
            psi_rad = psi_rad + 0.1;

        elif key == ord("z"):
            psi_rad = psi_rad - 0.1;

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

    print("Program Exit\n")
    print("############################################################\n")
