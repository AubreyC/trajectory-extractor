# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-21 18:51:37
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-08-06 12:41:05


import numpy as np
import os

from traj_ext.tracker import cameramodel
from traj_ext.utils import mathutil

# Define output dir for the test
DIR_PATH = os.path.dirname(__file__);
OUTPUT_DIR_PATH = os.path.join(DIR_PATH,'test_output');

def test_projection_ground():

    cam_matrix = np.array([[1920,   0, 1920/2],\
                            [  0, 1920, 1080/2],\
                            [  0,   0,      1]]);

    rot_CF_F = mathutil.eulerAnglesToRotationMatrix([np.deg2rad(60), 0 , 0]);
    trans_CF_F = rot_CF_F.dot(np.array([0,0,10]));

    # Define pixel:
    pixel_xy = np.array([np.random.randint(low=0, high=1920), np.random.randint(low=0, high=1080)]);

    pos_F = cameramodel.projection_ground(cam_matrix, rot_CF_F, trans_CF_F, 0, pixel_xy);
    pixel_xy_reproj = cameramodel.projection(cam_matrix, rot_CF_F, trans_CF_F, pos_F);

    # print("pixel_xy: {}\n".format(pixel_xy));
    # print("pixel_xy_reproj: {}\n".format(pixel_xy_reproj));

    assert (pixel_xy == pixel_xy_reproj).all()

def test_camera_model():

    # Camera Matrix
    cam_matrix_1 = np.array([[1280,    0, 1280/2],\
                               [  0, 1280,  720/2],\
                               [  0,    0,      1]], np.float64);

    rot_CF1_F = mathutil.eulerAnglesToRotationMatrix([1.32394204, -0.00242741, -0.23143667]);
    trans_CF1_F = np.array([22.18903449, -10.93100605, 78.07940989]);
    trans_CF1_F.shape = (3,1);

    dist_coeffs = np.zeros((4,1));
    cam_model = cameramodel.CameraModel(rot_CF1_F, trans_CF1_F, cam_matrix_1, dist_coeffs);

    # Define pixel:
    pixel_xy = np.array([np.random.randint(low=0, high=1280), np.random.randint(low=0, high=720)]);

    pos_F = cam_model.projection_ground(0.0, pixel_xy);

    pixel_xy_reproj = cam_model.project_points(pos_F);

    assert (pixel_xy == pixel_xy_reproj).all()

def test_save_camera():

    # rot_CF_F = np.identity(3);
    rot_CF_F = mathutil.eulerAnglesToRotationMatrix([0,0,np.pi/2]);

    trans_CF_F = np.array([0.0, 0.0, 186.0], dtype=float);
    trans_CF_F.shape = (3,1);

    focal_length = 1675.87;
    camera_matrix = np.zeros((3,3));
    camera_matrix[0,0] = focal_length;
    camera_matrix[1,1] = focal_length;
    camera_matrix[2,2] = 1;
    camera_matrix[0,2] = 1535.0/2.0;
    camera_matrix[1,2] = 876.0/2.0;
    dist_coeffs = np.zeros((4,1))

    # Save the parameters
    cam_model = cameramodel.CameraModel(rot_CF_F, trans_CF_F, camera_matrix, dist_coeffs);

    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    cam_model_path = os.path.join(OUTPUT_DIR_PATH, 'test_save_camera_calib.yml')
    cam_model.save_to_yml(cam_model_path);

    # Open camera parameters
    cam_model_2 = cameramodel.CameraModel.read_from_yml(cam_model_path);

    # Check for error:
    assert ((cam_model.cam_matrix == cam_model_2.cam_matrix).all()); #Could also use: np.array_equal(trans_CF_F, cam_model.trans_CF_F))
    assert ((cam_model.rot_CF_F == cam_model_2.rot_CF_F).all());
    assert ((cam_model.trans_CF_F == cam_model_2.trans_CF_F).all());
    assert ((cam_model.dist_coeffs == cam_model_2.dist_coeffs).all());

if __name__ == '__main__':
    test_projection_ground();
    test_camera_model();
    test_save_camera();