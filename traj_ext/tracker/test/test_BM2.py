# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-09-28 14:15:10
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-08-20 15:37:12

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
from traj_ext.tracker.cameramodel import CameraModel

from traj_ext.postprocess_track import trajectory

def get_mat_rot(alpha):


    R = np.array([[np.cos(alpha), -np.sin(alpha)],\
                  [np.sin(alpha),  np.cos(alpha)]])

    return R;

def make_box3D_from_pixel(cam_model, pix_meas_2D):

    box3D_init = [0.0, 0.0, 0.0, 0.0, 4.0, 2.0, -1.6];

    # Project pixel on the ground
    pos_FNED = cam_model.projection_ground(0, pix_meas_2D);
    pos_FNED.shape = (3,1);

    # Replace x,y,z in the default init box:
    box3D_init[1] = pos_FNED[0];
    box3D_init[2] = pos_FNED[1];
    box3D_init[3] = pos_FNED[2];

    return box3D_init;

def run_EKF_BM2(show_plot = False):

    cam_model = CameraModel.read_from_yml("traj_ext/camera_calib/calib_file/biloxi/biloxi_cam_cfg.yml");
    image_street = cv2.imread(os.path.join("traj_ext/camera_calib/calib_file/biloxi/biloxi_cam.jpg"));

    traj = trajectory.Trajectory(1, 'car');

    x_vel = -8.0*np.array([0.3,-1.0]);
    x_vel.shape = (2,1);
    dt = 0.1;

    x_pos = -0.1*np.array([-20.0,35.0]);
    x_pos.shape = (2,1);

    list_time_ms = [];
    for i in range(1000):

        time_ms = int(i*dt*1000);
        list_time_ms.append(time_ms);

        alpha = 0.1;
        if i > 200 and i < 500:
            alpha = -0.1;

        x_vel_ccurent = copy.copy(x_vel);

        if i > 20 and i < 60:
            x_vel_ccurent = 0.0*x_vel;
        else:
            R = get_mat_rot(alpha);
            x_vel = R.dot(x_vel);
            x_vel_ccurent = copy.copy(x_vel);


        x_pos = x_pos + x_vel_ccurent*dt;
        psi_rad = (np.arctan2(float(x_vel[1]),float(x_vel[0])));

        traj.add_point(time_ms, x_pos[0,0], x_pos[1,0], x_vel_ccurent[0,0], x_vel_ccurent[0,0], psi_rad);



    time_ms = list_time_ms[0];


    traj_point = traj.get_point_at_timestamp(time_ms);
    pt_pix = cam_model.project_points(np.array([(traj_point.x, traj_point.y, 0.0)]));

    # Create new traker:
    psi_rad = (np.arctan2(-float(traj_point.vy),float(traj_point.vx)));
    v_init = np.sqrt(traj_point.vx*traj_point.vx + traj_point.vy*traj_point.vy);
    x_init = np.array([traj_point.x,traj_point.y, v_init, psi_rad, 0], np.float64);
    x_init.shape = (5,1);

    P_init, Q, R = EKF_BM2.EKF_BM2_track.get_default_param();

    track_ekf = EKF_BM2.EKF_BM2_track( Q, R, P_init, 1, 'car');
    track_ekf.set_x_init(x_init, time_ms);

    pt_pix.shape = (2,1)
    track_ekf._push_pix_meas(time_ms, pt_pix);


    for index, time_ms in enumerate(list_time_ms[1:]):

        track_ekf.kf_predict(time_ms);

        # if not (index > 100 and index < 150):

        if index % 3 == 0:
            # Fuse measurement if there is one
            traj_point = traj.get_point_at_timestamp(time_ms);
            pt_pix = cam_model.project_points(np.array([(traj_point.x, traj_point.y, 0.0)]));

            pt_pix.shape = (2,1)

            # Fuse measurement:
            noise = np.random.normal(0, 10, 2);
            noise.shape = (2,1);
            pix_meas = pt_pix + noise;


            track_ekf.kf_fuse(pix_meas, cam_model, time_ms);



    print('Track_postprocess: Smoothing track_id: {}'.format(0));
    track_ekf.smooth(list_time_ms, post_proces = True);

    traj_smooth = trajectory.Trajectory(1, 'car', color = (0,0,255));

    for index, t_current_ms in enumerate(list_time_ms):
        # Getting info from EKF
        # TO DO: Clean this
        xy, vxy, phi_rad = track_ekf.get_processed_parameters_smoothed(t_current_ms);

        if not(xy is None):

            time_ms = int(t_current_ms);
            x = xy[0];
            y = xy[1];
            vx = vxy[0];
            vy = vxy[1];
            psi_rad = phi_rad;

            traj_smooth.add_point(time_ms, x, y, vx, vy, psi_rad);

    if show_plot:

        for index, time_ms in enumerate(list_time_ms):

            print('Index: {} time_ms: {}'.format(index, t_current_ms))

            image_current = copy.copy(image_street);

            image_current, _ = traj.display_on_image(time_ms, image_current, cam_model);
            image_current, _ = traj_smooth.display_on_image(time_ms, image_current, cam_model);

            pix_meas_2D = track_ekf.get_2Dpix_meas(time_ms);
            if not (pix_meas_2D is None):

                pix_meas = (int(pix_meas_2D[0]),int(pix_meas_2D[1]));
                image_current = cv2.circle(image_current, pix_meas,3, (255,0,255), -1);


            smooth_state = track_ekf.get_state_smooth_at_time(time_ms);
            print(smooth_state.P_cov)

            cv2.imshow('image_current', image_current)

            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break;

    return True;

# Dummy result of the test, only check if it runs.
def test_EKF_BM2():
    assert run_EKF_BM2();

# Main to run the test manually in case of failure
if __name__ == '__main__':
    run_EKF_BM2(True);