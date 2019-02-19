########################################################################################
#
# Implementation of utils function for Kalman filtering and smoothing script tracker_3D.py
#
########################################################################################

import cv2
import numpy as np
import sys
import os
import scipy.optimize as opt
import yaml
import argparse
import configparser
import csv

from utils import mathutil
from tracker import cameramodel as cm

def split(u, v, points):
    # return points on left side of UV
    return [p for p in points if np.cross(p[0:2] - u[0:2], v[0:2] - u[0:2]) < 0]

def extend(u, v, points):
    if not points:
        return []

    # find furthest point W, and split search to WV, UW
    w = min(points, key=lambda p: np.cross(p[0:2] - u[0:2], v[0:2] - u[0:2]))
    p1, p2 = split(w, v, points), split(u, w, points)
    return extend(w, v, p1) + [w] + extend(u, w, p2)

def convex_hull(points):
    # find two hull points, U, V, and split to left and right search
    u = min(points, key=lambda p: p[0])
    v = max(points, key=lambda p: p[0])
    left, right = split(u, v, points), split(v, u, points)

    # find convex hull on each side
    convex_full_list =  [v] + extend(u, v, left) + [u] + extend(v, u, right) + [v]
    return np.array(convex_full_list);

# Constraint function: Constraint the time to be positive
def func_cons(opti_params):
    focal_length = opti_params[0];
    return focal_length;


def func(opti_params, im_size, image_points, model_points_F):

    # Camera internals
    focal_length = opti_params[0]

    # Find camera parms
    rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs, image_points_reproj = find_camera_params(im_size, focal_length, image_points, model_points_F);

    # Compute error
    error_reproj = np.linalg.norm(np.subtract(image_points_reproj, image_points));

    return error_reproj;


def find_camera_params_opt(im_size, image_points, model_points_F, satellite_mode = False):

    # Pin-hole camera model: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    # Important to understand the Pin-Hole camera model:
    # s*[px; py; 1] = camera_matrix*[rot_CF_F, trans_CF_F]*[pos_F;1];
    # px, py: pixel position
    # rot_CF_F: frame rotation matrix from Camera Frame to Model Frame
    # trans_CF_F: Position of the origin of the Model Frame expressed in the Camera Frame
    # camera_matrix: intrinsic camera parameters
    # satellite_mode: Set the focal length to be size_image/2 since there is an ambiguity between focal_length and Z position for top-down view


    # Start with a guess:
    p_init = [im_size[1]];

    # Impose a constraint on the focal lenght: It must be positive
    cons = ({'type': 'ineq', 'fun': lambda x : func_cons(x)});

    # If not in satellite mode, also estimate the focal_lenght
    if not (satellite_mode):
        # Run the optimization to find the optimal parameters
        param = opt.minimize(func, p_init, constraints=cons, args=(im_size, image_points, model_points_F))
        focal_length = param.x[0]

    # Set the focal_lenght in sattelite mode to avoid ambiguity in the optimization
    else:
        focal_length = (im_size[0]);

    # Find camera parms
    rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs, image_points_reproj = find_camera_params(im_size, focal_length, image_points, model_points_F);

    # Compute error
    error_reproj = np.linalg.norm(np.subtract(image_points_reproj, image_points));

    # Ouput results
    if False:
        print("Error Reproj:\n {0}".format(error_reproj));
        print("Camera Matrix:\n {0}".format(camera_matrix));
        print("Rotation Vector:\n {0}".format(rot_vec_CF_F));
        print("Translation Vector:\n {0}".format(trans_CF_F));

    return rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs, image_points_reproj;

def find_camera_params(im_size, focal_length, image_points, model_points_F):

    # Pin-hole camera model: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    # Important to understand the Pin-Hole camera model:
    # s*[px; py; 1] = camera_matrix*[rot_CF_F, trans_CF_F]*[pos_F;1];
    # px, py: pixel position
    # rot_CF_F: frame rotation matrix from Camera Frame to Model Frame
    # trans_CF_F: Position of the origin of the Model Frame expressed in the Camera Frame
    # camera_matrix: intrinsic camera parameters

    # Camera internals
    focal_length = focal_length
    center = (im_size[1]/2, im_size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4,1))

    # Use solvePnP to compute the camera rotation and position from the 2D - 3D point corespondance
    (success, rot_vec_CF_F, trans_CF_F) = cv2.solvePnP(model_points_F, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Reproject model_points_F on image plane according to determined params
    imagePoints, jacobian = cv2.projectPoints(model_points_F, rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs)
    image_points_reproj = imagePoints[:,0]

    return rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs, image_points_reproj;


def convert_latlon_F(latlon_origin, latlon_points):

    model_points_F = np.array([]);
    model_points_F.shape = (0,3);

    for latlon_p in latlon_points:
        ned = mathutil.latlon_to_NED(latlon_origin, latlon_p);
        # Force to Z = 0
        ned = np.append(ned, 0);
        ned.shape = (1,3);
        model_points_F = np.append(model_points_F, ned, axis=0);

    return model_points_F;


def display_NED_frame(image, rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs):
    # Origin axis on image
    (pt_origin, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs)
    (pt_test_2, jacobian) = cv2.projectPoints(np.array([(10.0, 0.0, 0.0)]), rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs)
    (pt_test_3, jacobian) = cv2.projectPoints(np.array([(0.0, 10.0, 0.0)]), rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs)

    pt_o = ( int(pt_origin[0][0][0]), int(pt_origin[0][0][1]))
    pt_x = ( int(pt_test_2[0][0][0]), int(pt_test_2[0][0][1]))
    pt_y = ( int(pt_test_3[0][0][0]), int(pt_test_3[0][0][1]))

    # Add line of axis
    cv2.line(image, pt_o, pt_x, (255,0,0), 2)
    cv2.line(image, pt_o, pt_y, (255,255,0), 2)

    return image;

def display_keypoints(image, image_points_reproj, image_points):

    for i in range(0, image_points.shape[0]):
        cv2.circle(image, (int(image_points[i][0]), int(image_points[i][1])), 6, (0,0, 255), -1)

    # image_points_reproj
    for i in range(0, image_points_reproj.shape[0]):
        cv2.circle(image, (int(image_points_reproj[i][0]), int(image_points_reproj[i][1])), 6, (0,255, 255), -1)

    return image;


# Write camera calibration file:
def save_camera_calibration(output_path, rot_CF_F, trans_CF_F, camera_matrix, dist_coeffs):

    # Write Camera Param in YAML file
    fs_write = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
    fs_write.write('rot_CF_F', rot_CF_F);
    fs_write.write('trans_CF_F', trans_CF_F);
    fs_write.write('camera_matrix', camera_matrix);
    fs_write.write('dist_coeffs', dist_coeffs);
    fs_write.release()
    print('\n Camera config file saved %s \n' %(output_path))

# Read camera calibration file:
def read_camera_calibration(input_path):

    # Check input path
    if input_path == '':
        raise ValueError('[Error] Camera input path empty: {}'.format(input_path));

    # Intrinsic camera parameters
    fs_read = cv2.FileStorage(input_path, cv2.FILE_STORAGE_READ)
    cam_matrix = fs_read.getNode('camera_matrix').mat()
    rot_CF_F = fs_read.getNode('rot_CF_F').mat()
    trans_CF_F = fs_read.getNode('trans_CF_F').mat()
    dist_coeffs = fs_read.getNode('dist_coeffs').mat()

    # Some checks:
    if cam_matrix is None:
        raise ValueError('[Error] Camera cfg input path is wrong: {}'.format(input_path));

    # Construct camera model
    cam_model = cm.CameraModel(rot_CF_F, trans_CF_F, cam_matrix, dist_coeffs);

    return cam_model;


if __name__ == '__main__':

    points = np.random.rand(100, 2)
    hull = np.array(convex_hull(points))
    print(hull)

    # plot = figure()
    plt.plot(points[:, 0], points[:, 1], 'bo')
    plt.plot(hull[:, 0], hull[:, 1], 'ro')

    # # plot.plot(x=hull[:, 0], y=hull[:, 1], color='red')
    plt.show()

