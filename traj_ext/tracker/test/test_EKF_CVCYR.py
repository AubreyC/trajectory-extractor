# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-09-28 14:15:10
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-08-09 17:10:23

##########################################################################################
#
# Test of the EKF_CVCYR
#
##########################################################################################

import matplotlib
matplotlib.use('TkAgg')

import sys
import os

# FInd mathutils
from traj_ext.utils.mathutil import *

from traj_ext.tracker.EKF_CVCYR import *

def run_EKF_CVCYR(show_plot = False):

     # Process noise
    Q = np.array([[ 0.5,  0,   0,    0,     0],\
                   [ 0,  0.5,   0,    0,     0],\
                   [ 0,  0,     1,    0,     0],\
                   [ 0,  0,     0,    0.5,     0],\
                   [ 0,  0,     0,    0,     1]], np.float64);

    # Measurement noise: Pixel Noise
    R = np.array([[10,   0],\
                   [  0, 10]], np.float64);

    # Init Covariance
    P_init = np.array([[ 0.1,   0, 0,    0,    0],\
                        [ 0,   0.1, 0,    0,    0],\
                        [ 0,   0,    3,    0,    0],\
                        [ 0,   0,    0,    0.1,  0],\
                        [ 0,   0,    0,    0,    3]], np.float64);


    # # For ground truth:
    # Camera Matrix
    cam_matrix_1 = np.array([[1280,    0, 1280/2],\
                               [  0, 1280,  720/2],\
                               [  0,    0,      1]], np.float64);

    # rot_CF1_F: Frame rotation matrix from frame F to CameraFrame1 (street camera 1)
    rot_CF1_F = eulerAnglesToRotationMatrix([1.32394204, -0.00242741, -0.23143667]);
    trans_CF1_F = np.array([22.18903449, -10.93100605, 78.07940989]);
    trans_CF1_F.shape = (3,1);

    dist_coeffs = np.zeros((4,1));
    cam_model = cm.CameraModel(rot_CF1_F, trans_CF1_F, cam_matrix_1, dist_coeffs);

    # CAMERA_CFG_1_PATH = os.path.join(ROOT_DIR,'camera_calib/calib_file/auburn_camera_street_1_cfg.yml');
    # CAMERA_CFG_1_PATH = os.path.abspath(CAMERA_CFG_1_PATH);


    # fs_read = cv2.FileStorage(CAMERA_CFG_1_PATH, cv2.FILE_STORAGE_READ)
    # cam_matrix_1 = fs_read.getNode('camera_matrix').mat()
    # rot_CF1_F = fs_read.getNode('rot_CF_F').mat()
    # trans_CF1_F = fs_read.getNode('trans_CF_F').mat()
    # dist_coeffs_1 = fs_read.getNode('dist_coeffs').mat()

    # # Construct camera model
    # cam_model = cm.CameraModel(rot_CF1_F, trans_CF1_F, cam_matrix_1, dist_coeffs_1);


    # Generate fake data:
    # In NED
    x_i_tr = np.array([0, 20, 8, 1.5*(np.pi/3), 0], np.float64);
    x_i_tr.shape = (5,1)

    x_tr_log = np.array([]);
    x_tr_log.shape = (5,0);
    x_kf_log = np.array([]);
    x_kf_log.shape = (5,0);

    pix_tr_log = np.array([]);
    pix_tr_log.shape = (2,0);
    pix_meas_log = np.array([]);
    pix_meas_log.shape = (2,0);

    delta_s = np.float64(1)/15;
    x_c_tr = x_i_tr;
    x_i = copy.copy(x_i_tr);
    # x_i = np.array([0, 20, 0, 0, 0], np.float64);
    x_i.shape = (5,1);

    # Create new traker:
    t_current_s = 0;

    kf = EKF_CVCYR_track( Q, R, P_init, 0, 'car');
    kf.set_x_init(x_i, int(t_current_s*1e3));

    t_log = np.array([]);

    for i in range(0,100):

        # if i == 5:
        #     break;

        t_current_s = t_current_s + delta_s
        t_log = np.append(t_log, t_current_s);

        # Propagate ground truth
        pos_x   = x_c_tr[0,0];
        pos_y   = x_c_tr[1,0];
        v       = x_c_tr[2,0];
        phi     = wraptopi(x_c_tr[3,0]);
        phi_dot = x_c_tr[4,0];

        x_dot = np.array([[v*math.cos(phi),v*math.sin(phi), 0, phi_dot, 0]]);
        x_dot.shape = (5,1);

        # State predict
        x_c_tr.shape = (5,1);
        x_c_tr = x_c_tr + delta_s*x_dot;
        x_c_tr[3,0] = wraptopi(x_c_tr[3,0]);
        x_c_tr.shape = (5,1);

        pos_F = np.asarray(x_c_tr[0:2,0]).reshape(-1);
        pos_F = np.append(pos_F, 0);
        pix_tr = cam_model.project_points(pos_F)
        pix_tr.shape = (2,1);

        # For logging:
        pix_tr_log = np.append(pix_tr_log, pix_tr, axis = 1);

        x_tr_log = np.append(x_tr_log, x_c_tr,axis=1);
        x_kf_log = np.append(x_kf_log, kf.x_current, axis=1);

        # # Update KF:
        kf.kf_predict(float(t_current_s)*(1e3));

        # Fuse measurement:
        noise = np.random.normal(0, 2, 2);
        noise.shape = (2,1);
        pix_meas = pix_tr + noise;
        pix_meas_log = np.append(pix_meas_log, pix_meas, axis = 1);

        kf.kf_fuse(pix_meas, cam_model, float(t_current_s)*(1e3));

    print(x_kf_log.shape)
    print(np.asarray(x_tr_log[0,:]).reshape(-1))


    if show_plot:
        plt.figure()
        #Plot inverse Y axis because of the definition of the frame
        plt.plot(np.asarray(x_tr_log[0,:]).reshape(-1),np.asarray(x_tr_log[1,:]).reshape(-1));
        plt.plot(np.asarray(x_kf_log[0,:]).reshape(-1),np.asarray(x_kf_log[1,:]).reshape(-1));
        plt.ylabel('Position y (m)')
        plt.xlabel('Position x (m)')
        plt.xlim(-50,50)
        plt.ylim(100,0)
        plt.title('Position X Y')

        plt.figure()
        #Plot inverse Y axis because of the definition of the frame
        plt.plot(np.asarray(x_tr_log[0,:]).reshape(-1),np.asarray(x_tr_log[1,:]).reshape(-1));
        plt.plot(np.asarray(x_kf_log[0,:]).reshape(-1),np.asarray(x_kf_log[1,:]).reshape(-1));
        plt.ylabel('Position y (m)')
        plt.xlabel('Position x (m)')
        plt.xlim(-50,50)
        plt.ylim(100,0)
        plt.title('Position X Y')

        plt.figure()
        #Plot inverse Y axis because of the definition of the frame
        plt.plot(t_log,np.asarray(x_tr_log[3,:]).reshape(-1), label='True');
        plt.plot(t_log,np.asarray(x_kf_log[3,:]).reshape(-1), label='Estimate');
        plt.ylabel('Phi (rad)')
        plt.xlabel('Time (s)')
        plt.title('Phi')


        plt.figure()
        plt.plot(t_log,np.asarray(x_tr_log[2,:]).reshape(-1), label='True');
        plt.plot(t_log,np.asarray(x_kf_log[2,:]).reshape(-1), label='Estimate');
        plt.title('KF vel')
        plt.ylabel('Velocity (m/s)')
        plt.xlabel('time (s)')

        plt.figure()
        #Plot inverse Y axis because of the definition of the frame
        plt.plot(pix_tr_log[0,:], pix_tr_log[1,:], '.');
        plt.plot(pix_meas_log[0,:], pix_meas_log[1,:], 'x');
        plt.ylabel('Pix y (m)')
        plt.xlabel('Pix x (m)')
        plt.title('Image Camera')
        plt.xlim(0,1200)
        plt.ylim(1200,0)
        plt.show()

    return True;


# Dummy result of the test, only check if it runs.
def test_EKF_CVCYR():
    assert run_EKF_CVCYR();

# Main to run the test manually in case of failure
if __name__ == '__main__':
    run_EKF_CVCYR(True);
