# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-21 14:35:58
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-15 22:22:09

import os

from traj_ext.camera_calib import calib_utils
from traj_ext.utils import cfgutil
from traj_ext.utils import mathutil

from traj_ext.tracker.cameramodel import CameraModel

from traj_ext.camera_calib import run_calib_manual

import numpy as np
import cv2

# Define output dir for the test
DIR_PATH = os.path.dirname(__file__);
OUTPUT_DIR_PATH = os.path.join(DIR_PATH,'test_output');

def test_convex_hull(display = False):

    points = np.random.rand(100, 2)
    hull = np.array(calib_utils.convex_hull(points))
    print(hull)

    if display:

        plt.plot(points[:, 0], points[:, 1], 'bo')
        plt.plot(hull[:, 0], hull[:, 1], 'ro')

        # # plot.plot(x=hull[:, 0], y=hull[:, 1], color='red')
        plt.show()

def test_camera_calib():

    # Create camera model:
    rot_CF_F = mathutil.eulerAnglesToRotationMatrix([1.13,0.01,-2.22]);

    #-np.pi/2
    trans_CF_F = np.array([5.0, -2.6, 186.0], dtype=float);
    trans_CF_F.shape = (3,1);

    im_size = (1280,720);

    center = (im_size[1]/2, im_size[0]/2)

    focal_lenght = max(im_size);
    camera_matrix = np.array(
                             [[focal_lenght, 0, center[0]],
                             [0, focal_lenght, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    dist_coeffs = np.zeros((4,1))

    # Create camera model
    cam_model = CameraModel(rot_CF_F, trans_CF_F, camera_matrix, dist_coeffs);

    points_FNED = np.array([[ 10.,  10., 0.0],\
                            [-10., -10., 0.0],\
                            [ 10., -10., 0.0],\
                            [-10.,  10., 0.0]], dtype="double");

    image_points = np.zeros([1,2], dtype="double");
    for i in range(points_FNED.shape[0]):
        point_FNED = points_FNED[i,:];

        point_FNED.shape = (3,1);

        pt_current = cam_model.project_points(point_FNED);

        if i == 0:
            image_points[0,:] = (pt_current[0], pt_current[1]);

        else:

            d = np.array([[pt_current[0], pt_current[1]]], dtype= 'double');
            image_points = np.append( image_points, d , axis=0);


    rot_vec_CF_F_calib, trans_CF_F_calib, camera_matrix_calib, dist_coeffs, image_points_reproj = calib_utils.find_camera_params_opt(im_size, image_points, points_FNED);
    # Convert rotation vector in rotation matrix
    rot_CF_F_calib = cv2.Rodrigues(rot_vec_CF_F_calib)[0];
    print('rot_vec_CF_F_calib: ')
    print(rot_vec_CF_F_calib)

    # Convert rotation matrix in euler angle:
    euler_CF_F_calib = mathutil.rotationMatrixToEulerAngles(rot_CF_F_calib);
    print('euler_CF_F_calib: ')
    print(euler_CF_F_calib)

    print('euler_CF_F: ')
    print( mathutil.rotationMatrixToEulerAngles(rot_CF_F))

    # Position of the origin expresssed in CF
    print('trans_CF_F_calib (position of the origin expressed in CF): ')
    print(trans_CF_F_calib);

    print('trans_CF_F (position of the origin expressed in CF): ')
    print(trans_CF_F);

    print('camera_matrix_calib')
    print(camera_matrix_calib)

    print('camera_matrix')
    print(camera_matrix)


    print('image_points:\n {}'.format(image_points));
    print('image_points_reproj:\n {}'.format(image_points_reproj));

def test_camera_calib_from_csv():

    calib_points_path = os.path.join(DIR_PATH, 'brest_area1_street.csv');
    image_path = os.path.join(DIR_PATH,'brest_area1_street.jpg');

    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    run_calib_manual.run_calib_manual(calib_points_path, image_path, False, OUTPUT_DIR_PATH, auto_save = True);

if __name__ == '__main__':
    test_camera_calib();
    test_convex_hull();
    test_camera_calib_from_csv();
