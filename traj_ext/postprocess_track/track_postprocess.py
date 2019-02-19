# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-11-26 17:45:38
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

########################################################################################
#
# Implementation of an Extended Kalman Filter for Vehicle Tracking from Image Detection
# Parent Class
#
########################################################################################


import numpy as np;
import matplotlib.pyplot as plt
from  math import *;
import os
import cv2
import copy
import sys
import time
import abc

import trajectory

ROOT_DIR = os.getcwd()

sys.path.append('../tracker')
import cameramodel as cm

sys.path.append('../utils')
from mathutil import *
import cfgutil

sys.path.append(os.path.join(ROOT_DIR,'../tracker/utils/'))
import tracker_utils

class MeasStruct():
    def __init__(self, time_ms, pix_meas_2D, label):
        self.time_ms = time_ms;
        self.pix_meas_2D = pix_meas_2D;
        self.label = label;

        # Making sure pix_meas is the right format / size
        if not (pix_meas_2D.shape == (2,1)):
            raise NameError('MeasStruct size is not correct')

class MeasBox3D():
    def __init__(self, time_ms, box3D, label):
        self.time_ms = time_ms;
        self.box3D = box3D;
        self.label = label;

class Track_postprocess(object):

    def __init__(self, track_id, cam_model_meas, dynamic_model = 'CV'):

        # Tracker ID:
        self._track_id = track_id
        self._dynamic_model = dynamic_model;
        self._cam_model_meas = cam_model_meas;

        # Box3D Init:
        self._box3D_init = [0.0, 0.0, 0.0, 0.0, 4.0, 2.0, -1.6];
        self._box3D_init_set = False;

        # Store past measurement
        self._meas_list = [];
        self._box3D_meas_list = [];

        self._track_ekf = None;
        self._trajectory = None;


    def get_trajectory(self):
        return self._trajectory;

    def push_pix_meas(self, time_ms, pix_meas_2D, label):

        meas_data = MeasStruct(time_ms, pix_meas_2D, label);

        if len(self._meas_list) > 0:
            delta_ms = time_ms - self._meas_list[-1].time_ms;
            if delta_ms > 1000:
                print('[WARNING]: meas list: {}'.format(self._meas_list[-1].time_ms))
                print('[WARNING]: Measurement has {}ms interval for track_id: {}'.format(delta_ms, self._track_id))

        self._meas_list.append(meas_data);

        # Set the initial box 3D:
        if not (self._box3D_init_set):
            # Project pixel on the ground
            pos_FNED = self._cam_model_meas.projection_ground(0, pix_meas_2D);
            pos_FNED.shape = (3,1);

            # Replace x,y,z in the default init box:
            self._box3D_init[1] = pos_FNED[0];
            self._box3D_init[2] = pos_FNED[1];
            self._box3D_init[3] = pos_FNED[2];

            # Set the init flag
            self._box3D_init_set = True;

        return;

    def compute_highest_label(self):

        label_list = [];
        for pix_meas in self._meas_list:
            label_list.append(pix_meas.label);

        max_label = cfgutil.compute_highest_occurence(label_list);

        return max_label;

    def push_3Dbox_meas(self, time_ms, box3D_meas, label):

        if box3D_meas is None:
            print('[Error] : Track_postprocess push_3Dbox_meas box3D_meas is None')
            return;

        # Add to 3D box meas list:
        box3D_meas_data = MeasBox3D(time_ms, box3D_meas, label);
        self._box3D_meas_list.append(box3D_meas_data);

        # Set the initial box 3D:
        if not (self._box3D_init_set):
            self._box3D_init = box3D_meas;
            self._box3D_init_set = True;

        # Project on Image
        pt_F = np.array(box3D_meas[1:4]);
        pt_F.shape = (3,1)

        (pt_img, jacobian) = cv2.projectPoints(pt_F.transpose(), self._cam_model_meas.rot_CF_F, self._cam_model_meas.trans_CF_F, self._cam_model_meas.cam_matrix, self._cam_model_meas.dist_coeffs)
        pt_img_tulpe = (int(pt_img[0][0][0]), int(pt_img[0][0][1]));

        pix_meas_2D = np.array([int(pt_img[0][0][0]), int(pt_img[0][0][1])]);
        pix_meas_2D.shape = (2,1);

        # Add to pix meas list
        self.push_pix_meas(time_ms, pix_meas_2D, label);

        return;

    def get_id(self):
        return self._track_id;

    def _create_list_time(self, full_times_ms, init_time_ms, end_time_ms):

        try:
            i_start = full_times_ms.index(init_time_ms);
            i_end = full_times_ms.index(end_time_ms);
        except Exception as e:
            print("[ERROR]: Track_postprocess _create_list_time init_time_ms or end_time_ms not found");
            raise

        l_times_ms = full_times_ms[i_start:i_end + 1];

        return l_times_ms;

    def _get_index_meas(self, time_ms):

        index = None;
        for i, meas in enumerate(self._meas_list):
            if meas.time_ms == time_ms:
                index = i;
                break;

        return index;

    def _get_index_box3D_meas(self, time_ms):

        index = None;
        for i, meas in enumerate(self._box3D_meas_list):
            if meas.time_ms == time_ms:
                index = i;
                break;

        return index;


    def get_tracker_EKF(self):
        return self._track_ekf;

    def process_traj(self, list_times_ms):

        # Making sure list of measurement is not empty
        if not(len(self._meas_list) > 1):
            return;

        # Create list time for this traks
        init_time_ms = self._meas_list[0].time_ms;
        end_time_ms = self._meas_list[-1].time_ms;
        l_times_ms = self._create_list_time(list_times_ms, init_time_ms, end_time_ms);

        t_current_ms = l_times_ms[0];

        # Create new traker:
        self._track_ekf = tracker_utils.create_tracker(self._dynamic_model , self._box3D_init, self._track_id, t_current_ms, 'car');
        self._track_ekf.push_3Dbox_meas(t_current_ms,self._box3D_init);

        # =========================================
        # Filtering
        # ==========================================
        print('Track_postprocess: Filtering track_id: {}'.format(self._track_id));

        # Skip first time as init already done:
        for t_current_ms in l_times_ms[1:]:

            # Predict
            self._track_ekf.kf_predict(t_current_ms);

            # Fuse measurement if there is one
            index_meas = self._get_index_meas(t_current_ms);
            if not (index_meas is None):

                pix_meas = self._meas_list[index_meas].pix_meas_2D;
                self._track_ekf.kf_fuse(pix_meas, self._cam_model_meas, t_current_ms);

                index_box3D_meas = self._get_index_box3D_meas(t_current_ms);
                if not(index_box3D_meas is None):
                    self._track_ekf.push_3Dbox_meas(t_current_ms, self._box3D_meas_list[index_box3D_meas].box3D);
        # =========================================
        # Smoothing
        # ==========================================
        print('Track_postprocess: Smoothing track_id: {}'.format(self._track_id));
        self._track_ekf.smooth(l_times_ms, post_proces = True);

        # Compute label of the trajectory - max occurence label
        max_label = self.compute_highest_label();

        # =========================================
        # Generate a trajectory
        # ==========================================
        self._trajectory = trajectory.Trajectory(self._track_id, max_label, self._track_ekf.get_color());

        for index, t_current_ms in enumerate(l_times_ms):
            # Getting info from EKF
            # TO DO: Clean this
            xy, vxy, phi_rad = self._track_ekf.get_processed_parameters_smoothed(t_current_ms);

            if not(xy is None):

                time_ms = int(t_current_ms);
                x = xy[0];
                y = xy[1];
                vx = vxy[0];
                vy = vxy[1];
                psi_rad = phi_rad;

                self._trajectory.add_point(time_ms, x, y, vx, vy, psi_rad);

        # =========================================
        # TEMPORARY: Re-filter the psi angle:
        # ==========================================


        if self._dynamic_model == 'CV':
            traj = self._trajectory.get_traj();

            psi_rad_first = None;
            for index in range(len(traj)):
                if not(traj[index].psi_rad is None):
                    psi_rad_first = traj[index].psi_rad;
                    break;

            if psi_rad_first is None:
                psi_rad_first = 0;

            psi_rad_last = psi_rad_first;
            for index in range(len(traj)):

                # if psi_rad_last is None:
                #     psi_rad_last = traj[index].psi_rad;

                v = np.sqrt(np.square(traj[index].vx) + np.square(traj[index].vy));
                if v < 0.8:
                    traj[index].psi_rad = psi_rad_last;
                    print('Kepp last psi'.format(self._track_id))

                psi_rad_last = traj[index].psi_rad;

                if traj[index].psi_rad is None:
                    raise NameError('traj[index].psi_rad None index: {}'.format(index))

            self._trajectory._traj = traj;