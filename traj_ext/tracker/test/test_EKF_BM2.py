# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-09-28 14:15:10
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-08-09 17:09:11

##########################################################################################
#
# Test of the EKF_BM2
#
##########################################################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sys
import os
import cv2
import copy

# FInd mathutils
from traj_ext.utils.mathutil import *
from traj_ext.tracker import EKF_BM2
from traj_ext.tracker import cameramodel as cm

# In 2D, display the ellispe corresponding to the covariance matrix
def compute_ellipse(ax, chisquare_val, pt_xy, mat_cov):
        #From: https://gist.github.com/eroniki/2cff517bdd3f9d8051b5
        # Inputs:
        # img: img to draw the ellispe on
        # chisquare_val: confidence indice since e normalized follow a Chi-Sqaured distribution
        # pt_xy: center of the ellispe
        # mat_cov: Covaraince matrix

    # Compute eigenvalues and eigenvectors
    e_val, e_vec = np.linalg.eig(mat_cov);

    # Reorder to get max eigen first
    # if e_val[1] > e_val[1]:
    #     e_val = reverse(e_val);
    #     e_vec = reverse(e_vec);

    # Compute angle between eigenvector and x axis
    angle = -np.arctan2(e_vec[0,1],e_vec[0,0]);

    # Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
    if angle < 0 :
        angle += 6.28318530718;

    halfmajoraxissize = (chisquare_val*np.sqrt(math.fabs(e_val[0])));
    halfminoraxissize = (chisquare_val*np.sqrt(math.fabs(e_val[1])));

    # print(halfmajoraxissize);
    # print(halfminoraxissize);
    e1 = Ellipse(pt_xy, halfmajoraxissize, halfminoraxissize, angle=np.rad2deg(angle), linewidth=1, fill=False, zorder=2)
    ax.add_patch(e1)

    # ellipse = Ellipse(pt_xy, halfmajoraxissize, halfminoraxissize, angle)  # color="k")
    # ellipse.set_clip_box(ax.bbox)
    # ellipse.set_alpha(0.2)
    # ax.add_artist(ellipse)

    #print (halfmajoraxissize, halfminoraxissize)
    #cv2.ellipse(img,pt_xy,(halfmajoraxissize, halfminoraxissize), angle,0,360,255, 2)
    # cv2.ellipse(img,pt_xy,(halfmajoraxissize, halfminoraxissize), np.rad2deg(angle),0,360,255, 2)


def plot_ellipse(ax, mu, sigma, color="b"):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(sigma)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)
    ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)

def run_EKF_BM2(show_plot = False):

     # Process noise
    Q = np.array([[  0,  0,     0,    0,     0],\
                   [ 0,  0,     0,    0,     0],\
                   [ 0,  0,     0.5,    0,     0],\
                   [ 0,  0,     0,    0,     0],\
                   [ 0,  0,     0,    0,     0.001]], np.float64);

    # Measurement noise: Pixel Noise
    R = np.array([[25,   0],\
                   [  0, 25]], np.float64);

    # Init Covariance
    P_init = np.array([[   0,     0,    0,    0,    0],\
                        [   0,     0,    0,    0,    0],\
                        [   0,     0,    1,    0,    0],\
                        [   0,     0,    0,    2,    0],\
                        [   0,     0,    0,    0,     0.01]], np.float64);


    # For ground truth:
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

    # Generate fake data:
    # In NED
    x_i_tr = np.array([0, 20, 2, np.pi/2, 0], np.float64);
    x_i_tr.shape = (5,1)

    P_kf_log = np.array([]);
    P_kf_log.shape = (5,0);

    P_kf_full_log = [];

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

    t_current_s = 0;

    # Create new traker:
    kf = EKF_BM2.EKF_BM2_track( Q, R, P_init, 0, 'car');
    kf.set_x_init(x_i, int(t_current_s*1e3));

    t_log = np.array([]);

    for i in range(0,150):

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

        if not (i > 20 and i < 80):
            # if i%2 == 0:
            kf.kf_fuse(pix_meas, cam_model, float(t_current_s)*(1e3));

        P_current = kf.P_current;
        P_current_x = np.array([P_current[0,0], P_current[1,1], P_current[2,2], P_current[3,3], P_current[4,4]]);
        P_current_x.shape = (5,1);
        P_kf_log = np.append(P_kf_log, P_current_x, axis=1);

        P_kf_full_log.append(P_current);

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
        plt.subplot(2, 1, 1)
        #Plot inverse Y axis because of the definition of the frame
        plt.plot(t_log,np.asarray(x_tr_log[3,:]).reshape(-1), label='True');
        plt.plot(t_log,np.asarray(x_kf_log[3,:]).reshape(-1),'*');
        plt.ylabel('Phi (rad)')
        plt.xlabel('Time (s)')
        plt.title('Phi')

        plt.subplot(2, 1, 2)
        #Plot inverse Y axis because of the definition of the frame
        plt.plot(t_log,np.asarray(x_tr_log[4,:]).reshape(-1), label='True');
        plt.plot(t_log,np.asarray(x_kf_log[4,:]).reshape(-1),'*');
        plt.ylabel('Beta (rad)')
        plt.xlabel('Time (s)')
        plt.title('Betaq')

        plt.figure()
        plt.plot(t_log,np.asarray(x_tr_log[2,:]).reshape(-1), label='True');
        plt.plot(t_log,np.asarray(x_kf_log[2,:]).reshape(-1), '*');
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

        plt.figure()
        plt.subplot(5, 1, 1)
        plt.plot(t_log,np.asarray(P_kf_log[0,:]).reshape(-1), label='Estimate');
        plt.title('Test: KF P x')
        plt.ylabel('')
        plt.xlabel('time (s)')

        plt.subplot(5, 1, 2)
        plt.plot(t_log,np.asarray(P_kf_log[1,:]).reshape(-1), label='Estimate');
        plt.title('Test: KF P y')
        plt.ylabel('')
        plt.xlabel('time (s)')

        plt.subplot(5, 1, 3)
        plt.plot(t_log,np.asarray(P_kf_log[2,:]).reshape(-1), label='Estimate');
        plt.title('Test: KF P v')
        plt.ylabel('')
        plt.xlabel('time (s)')

        plt.subplot(5, 1, 4)
        plt.plot(t_log,np.asarray(P_kf_log[3,:]).reshape(-1), label='Estimate');
        plt.title('Test: KF P psi')
        plt.ylabel('')
        plt.xlabel('time (s)')

        plt.subplot(5, 1, 5)
        plt.plot(t_log,np.asarray(P_kf_log[4,:]).reshape(-1), label='Estimate');
        plt.title('Test: KF P beta')
        plt.ylabel('')
        plt.xlabel('time (s)')


        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='auto')
        # ax.plot(np.asarray(x_kf_log[0,:]).reshape(-1),np.asarray(x_kf_log[1,:]).reshape(-1), '*');
        ax.set_xlim(-50, 50)
        ax.set_ylim(100, 0)

        for ind,P in enumerate(P_kf_full_log):
            if ind%5 == 0:
                    xy = x_kf_log[0:2,ind];
                    compute_ellipse(ax, 2.7, xy, P[0:2, 0:2]);
                    ax.plot(xy[0], xy[1], '*');



        # #Plot inverse Y axis because of the definition of the frame
        # plt.plot(np.asarray(x_tr_log[0,:]).reshape(-1),np.asarray(x_tr_log[1,:]).reshape(-1));
        # plt.plot(np.asarray(x_kf_log[0,:]).reshape(-1),np.asarray(x_kf_log[1,:]).reshape(-1));
        # plt.ylabel('Position y (m)')
        # plt.xlabel('Position x (m)')
        # plt.xlim(-50,50)
        # plt.ylim(100,0)
        # plt.title('Position X Y')


        plt.show()

    return True;


# Dummy result of the test, only check if it runs.
def test_EKF_BM2():
    assert run_EKF_BM2();

# Main to run the test manually in case of failure
if __name__ == '__main__':
    run_EKF_BM2(True);
