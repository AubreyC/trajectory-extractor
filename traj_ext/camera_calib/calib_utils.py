########################################################################################
#
# Implementation of utils function for camera calibration
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

from traj_ext.utils import mathutil
from traj_ext.tracker import cameramodel as cm

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


    # points_F_list = [];
    # for i in range(model_points_F.shape[0]):
    #     # print(model_points_F[i,:])
    #     # print(model_points_F.shape[0])
    #     points_F_list.append(model_points_F[i,:]);

    # points_px_list = [];
    # for i in range(image_points.shape[0]):
    #     # print(image_points[i,:])
    #     # print(image_points.shape[0])
    #     points_px_list.append(image_points[i,:]);

    # print(points_px_list)
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([np.float32([points_F_list])], [np.float32([points_px_list])], im_size,None,None)
    # print('mtx: {}'.format(mtx))
    # print('rvecs: {}'.format(rvecs))
    # print('tvecs: {}'.format(tvecs))

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

def display_keypoints(image, image_points_reproj, image_points):

    for i in range(0, image_points.shape[0]):
        cv2.circle(image, (int(image_points[i][0]), int(image_points[i][1])), 6, (0,0, 255), -1)

    # image_points_reproj
    for i in range(0, image_points_reproj.shape[0]):
        cv2.circle(image, (int(image_points_reproj[i][0]), int(image_points_reproj[i][1])), 6, (0,255, 255), -1)

    return image;