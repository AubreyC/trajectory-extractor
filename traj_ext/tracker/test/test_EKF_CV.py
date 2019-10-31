# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-09-28 14:15:10
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-08-09 17:09:49

##########################################################################################
#
# Test of the EKF_CV
#
##########################################################################################
import matplotlib
matplotlib.use('TkAgg')

import sys
import os

# Find mathutils
from traj_ext.utils.mathutil import *

from traj_ext.tracker.EKF_CV import *

def run_EKF_CV(show_plot = False):

    # Process noise
    Q = np.array([[ 0,  0,    0,    0],\
                   [ 0,  0,    0,    0],\
                   [ 0,  0,  0.0001,    0],\
                   [ 0,  0,    0,  0.0001]]);

    # Measurement noise: Pixel Noise
    R = np.array([[ 80,  0],\
                   [  0, 80]]);

    # Init Covariance
    P_init = np.array([[ 0,  0,    0,    0],\
                        [ 0,  0,    0,    0],\
                        [ 0,  0,    10,    0],\
                        [ 0,  0,    0,    10]]);


    # For ground truth:
    # Camera Matrix
    cam_matrix_1 = np.array([[1280,    0, 1280/2],\
                               [  0, 1280,  720/2],\
                               [  0,    0,      1]]);

    # rot_CF1_F: Frame rotation matrix from frame F to CameraFrame1 (street camera 1)
    rot_CF1_F = eulerAnglesToRotationMatrix([1.32394204, -0.00242741, -0.23143667]);
    trans_CF1_F = np.array([22.18903449, -10.93100605, 78.07940989]);
    trans_CF1_F.shape = (3,1);

    dist_coeffs = np.zeros((4,1));
    cam_model = cm.CameraModel(rot_CF1_F, trans_CF1_F, cam_matrix_1, dist_coeffs);

    # Generate fake data:
    # In NED
    x_i_tr = np.array([0,0,-0.2,1])

    x_tr_log = np.array([]);
    x_tr_log.shape = (4,0);
    x_kf_log = np.array([]);
    x_kf_log.shape = (4,0);

    P_kf_log = np.array([]);
    P_kf_log.shape = (4,0);


    pix_tr_log = np.array([]);
    pix_tr_log.shape = (2,0);
    pix_meas_log = np.array([]);
    pix_meas_log.shape = (2,0);

    delta_s = float(1)/15;
    x_c_tr = x_i_tr;
    x_i = np.array([0,0,-0.2,1])
    x_i.shape = (4,1);

    # Init Kalman Filter:
    t_current_s = 0;

    kf = EKF_CV_track(Q, R, P_init, 0, 'a');
    kf.set_x_init(x_i, int(t_current_s*1e3));

    t_log = np.array([]);

    for i in range(0,1000):

        t_current_s = t_current_s + delta_s
        t_log = np.append(t_log, t_current_s);

        # Generate Ground Truth:
        F = np.identity(4) + kf.A*delta_s;
        x_c_tr.shape = (4,1);
        x_c_tr = F.dot(x_c_tr);

        pos_F = np.asarray(x_c_tr[0:2,0]).reshape(-1);
        pos_F = np.append(pos_F, 0);
        pix_tr = cam_model.project_points(pos_F)
        pix_tr.shape = (2,1);

        # For logging:
        pix_tr_log = np.append(pix_tr_log, pix_tr, axis = 1);

        x_tr_log = np.append(x_tr_log, x_c_tr,axis=1);
        x_kf_log = np.append(x_kf_log, kf.x_current, axis=1);

        P_current = kf.P_current;
        P_current_x = np.array([P_current[0,0], P_current[1,1], P_current[2,2], P_current[3,3]]);
        P_current_x.shape = (4,1);
        P_kf_log = np.append(P_kf_log, P_current_x, axis=1);

        # # Update KF:
        kf.kf_predict(float(t_current_s)*float(1e3));

        # Fuse measurement:
        noise = np.random.normal(0, 0.1, 2);
        noise.shape = (2,1);
        pix_meas = pix_tr;
        pix_meas_log = np.append(pix_meas_log, pix_meas, axis = 1);

        if not (i > 500 and i < 600):
            # if i%2 == 0:
            kf.kf_fuse(pix_meas, cam_model, float(t_current_s)*float(1e3));


    print("Result:")
    print(np.asarray(x_tr_log[0,:]).reshape(-1))

    if(show_plot):
        plt.figure()
        #Plot inverse Y axis because of the definition of the frame
        plt.plot(np.asarray(x_tr_log[0,:]).reshape(-1),np.asarray(x_tr_log[1,:]).reshape(-1));
        plt.plot(np.asarray(x_kf_log[0,:]).reshape(-1),np.asarray(x_kf_log[1,:]).reshape(-1));
        plt.ylabel('Position y (m)')
        plt.xlabel('Position x (m)')
        plt.xlim(-50,50)
        plt.ylim(100,0)
        plt.title('Test: KF')

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t_log,np.asarray(x_tr_log[2,:]).reshape(-1), label='True');
        plt.plot(t_log,np.asarray(x_kf_log[2,:]).reshape(-1), label='Estimate');
        plt.title('Test: KF vel')
        plt.ylabel('Velocity x (m/s)')
        plt.xlabel('time (s)')

        plt.subplot(2, 1, 2)
        plt.plot(t_log,np.asarray(x_tr_log[3,:]).reshape(-1), label='True');
        plt.plot(t_log,np.asarray(x_kf_log[3,:]).reshape(-1), label='Estimate');
        plt.title('Test: KF vel')
        plt.ylabel('Velocity y (m/s)')
        plt.xlabel('time (s)')
        plt.legend()

        plt.figure()
        plt.subplot(4, 1, 1)
        plt.plot(t_log,np.asarray(P_kf_log[0,:]).reshape(-1), label='Estimate');
        plt.title('Test: KF P x')
        plt.ylabel('')
        plt.xlabel('time (s)')

        plt.subplot(4, 1, 2)
        plt.plot(t_log,np.asarray(P_kf_log[1,:]).reshape(-1), label='Estimate');
        plt.title('Test: KF P y')
        plt.ylabel('')
        plt.xlabel('time (s)')

        plt.subplot(4, 1, 3)
        plt.plot(t_log,np.asarray(P_kf_log[2,:]).reshape(-1), label='Estimate');
        plt.title('Test: KF P vx')
        plt.ylabel('')
        plt.xlabel('time (s)')

        plt.subplot(4, 1, 4)
        plt.plot(t_log,np.asarray(P_kf_log[3,:]).reshape(-1), label='Estimate');
        plt.title('Test: KF P vy')
        plt.ylabel('')
        plt.xlabel('time (s)')

        plt.figure()
        #Plot inverse Y axis because of the definition of the frame
        plt.plot(pix_tr_log[0,:], pix_tr_log[1,:], '.');
        plt.plot(pix_meas_log[0,:], pix_meas_log[1,:], 'x');
        plt.ylabel('Pix y (m)')
        plt.xlabel('Pix x (m)')
        plt.title('Test: KF')
        plt.xlim(0,1200)
        plt.ylim(1200,0)
        plt.show()

    return True;


# Dummy result of the test, only check if it runs.
def test_EKF_CV():
    assert run_EKF_CV();

# Main to run the test manually in case of failure
if __name__ == '__main__':
    run_EKF_CV(True);
