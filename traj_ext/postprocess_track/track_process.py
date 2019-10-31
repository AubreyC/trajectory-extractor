# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-08-13 13:57:11
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-08-21 15:12:39

import numpy as np;
import matplotlib.pyplot as plt
from  math import *;
import os
import cv2
import copy
import sys
import time
import abc
from enum import Enum

from traj_ext.det_association.track_2D import *

from traj_ext.postprocess_track import trajectory
from traj_ext.utils import det_zone


from traj_ext.utils.mathutil import *
from traj_ext.utils import cfgutil

from traj_ext.tracker import EKF_utils
from traj_ext.tracker import EKF_CV
from traj_ext.tracker import EKF_BM2

class ProjectionMode(Enum):
    BOX3D = 1;
    CENTER_2D = 2;
    CENTER_2D_HEGHT = 3;

class TrackProcess(object):
    """TrackProcess process a track2D with Kalman Filter / RTS smoother to obtain trajectory"""

    # Projection height used with mode_2D_3D = 'center_2d_height'
    HEIGHT_PROJECTION = {'person':      -0.6,\
                         'car':         -0.6,\
                         'bus':         -1.5,\
                         'truck':       -1.0,\
                         'bicycle':     -0.6,\
                         'motorcycle':  -0.6};

    def __init__(self, track_2D, cam_model_meas, dynamic_model = 'CV',  projection_mode = 'center_2d_height', box3D_minimum_overlap = 0.0):

        # Track 2D holding detections
        self.track_2D = track_2D;

        # Projection mode
        if projection_mode == 'center_2d_height':
            self.projection_mode = ProjectionMode.CENTER_2D_HEGHT;
        elif projection_mode == 'center_2d':
            self.projection_mode = ProjectionMode.CENTER_2D;
        elif projection_mode == 'box3D':
            self.projection_mode = ProjectionMode.BOX3D;
        else:
            raise NameError('[TrackProcess]: ERROR Unknown projection mode: {}'.format(projection_mode));

        # Dynamical model used for smoothing
        self.dynamic_model = dynamic_model;
        self.cam_model_meas = cam_model_meas;
        self.track_ekf = None;

        # Trajectory object
        self.trajectory = None;

        # Set agent type from track 2D
        self.agent_type = track_2D.agent_type;

        # Minimum overlap for box3D if projection_mode box3D
        self.box3D_minimum_overlap = box3D_minimum_overlap;

    def get_pos_FNED_from_frame_index(self, frame_index):
        """Compute the FNED position from a measurment at frame_index

        Args:
            frame_index (TYPE): Frame Index

        Returns:
            TYPE: Position in FNED
        """
        # 3D position by re-projection on the ground
        pix_meas = self.get_pix_meas_from_index(frame_index, no_box_overlap_check = True);

        pt_image = (int(pix_meas[0,0]), int(pix_meas[1,0]))
        pos_FNED = self.cam_model_meas.projection_ground(0, pt_image);
        pos_FNED.shape = (3,1);

        return pos_FNED;

    def get_pix_meas_from_index(self, frame_index, no_box_overlap_check = False):
        """Compute the pixel measurement for a frame_index, based on mode_2D_3D.

        Args:
            frame_index (TYPE): Frame Index

        Returns:
            TYPE: Pixel Measurement
        """

        # Get detction object for frame_index
        meas = self.track_2D.get_meas_frame_index(frame_index);

        pix_meas = None;
        if not (meas is None):

            # Pixel position from the Bottom Center of the 3D box estimated with 3D fitting process
            if self.projection_mode == ProjectionMode.BOX3D:

                if not (meas.box3D is None):

                    # Only if percent_overlap is higher than self.box3D_minimum_overlap
                    if no_box_overlap_check or meas.box3D.percent_overlap > self.box3D_minimum_overlap:
                        pix_meas = meas.box3D.project_box_bottom_center_image(self.cam_model_meas);

                else:
                    print('[TrackProcess] WARNING meas.box3D is None track_id: {} frame_index: {} det_id: {}'.format(self.get_id(), frame_index, meas.det_obj.det_id))

            # Pixel measurement directly center of detection
            elif self.projection_mode == ProjectionMode.CENTER_2D:

                # Get det object from meas object
                det_2D = meas.det_obj;
                if det_2D is None:
                    raise NameError('[TrackProcess] WARNING meas.det_obj is None track_id: {} frame_index: {}'.format(self.get_id(), frame_index))

                # Get center of detection
                pix_center_2D = det_2D.get_center_det_2Dbox();
                pix_center_2D = np.array(pix_center_2D);
                pix_center_2D.shape = (2,1);

                pix_meas = pix_center_2D;

            # Pixel measurement: Center of detection is at some height above the ground, compute the pixel location of the actual ground position
            elif self.projection_mode == ProjectionMode.CENTER_2D_HEGHT:

                # Get det object from meas object
                det_2D = meas.det_obj;
                if det_2D is None:
                    raise NameError('[TrackProcess] WARNING meas.det_obj is None track_id: {} frame_index: {}'.format(self.get_id(), frame_index))

                # Get center of detection
                pix_center_2D = det_2D.get_center_det_2Dbox();
                pix_center_2D = np.array(pix_center_2D);
                pix_center_2D.shape = (2,1);

                height_proj = 0.0;
                if self.agent_type in self.HEIGHT_PROJECTION.keys():

                    height_proj = self.HEIGHT_PROJECTION[self.agent_type]

                else:
                    raise NameError ('[TrackProcess]: ERROR get_pix_meas_from_index self.agent_type: {} not in HEIGHT_PROJECTION: {}'.format(self.agent_type, self.HEIGHT_PROJECTION.keys()))

                pos_FNED = self.cam_model_meas.projection_ground(height_proj, pix_center_2D);
                pos_FNED[2] = 0.0;

                pix_meas_new = self.cam_model_meas.project_points(pos_FNED);
                pix_meas_new.shape = (2,1);

                pix_meas = pix_meas_new;

            else:
                raise NameError ('[TrackProcess]: ERROR Unknown projection_mode: {}'.format(self.projection_mode))

        return pix_meas;

    def set_agent_type(self, agent_type):
        """Set agent type

        Args:
            agent_type (TYPE): Description
        """

        self.agent_type = agent_type;

    def get_id(self):
        """Get track ID

        Returns:
            TYPE: Track ID
        """
        return self.track_2D.track_id;

    def get_trajectory(self):
        """Get the trajectory of this track

        Returns:
            TYPE: Trajectory
        """
        return self.trajectory;

    def create_list_time(self, full_times_ms, init_time_ms, end_time_ms):

        try:
            i_start = full_times_ms.index(init_time_ms);
            i_end = full_times_ms.index(end_time_ms);
        except Exception as e:
            raise NameError("[ERROR]: Track_postprocess _create_list_time init_time_ms or end_time_ms not found");

        l_times_ms = full_times_ms[i_start:i_end + 1];

        return l_times_ms;


    def process_traj(self, list_times_ms, reverse_init_BM = False):

        # Making sure list of measurement is not empty
        if not(self.track_2D.get_length() > 0):
            print('Trak Process {}: only {}'.format(self.get_id(), self.track_2D.get_length()))
            return;

        # Get init and last frame index
        init_frame_index = self.track_2D.get_init_frame_index();
        last_frame_index = self.track_2D.get_last_frame_index();

        # Create list time for this traks
        init_time_ms = list_times_ms[init_frame_index];
        end_time_ms = list_times_ms[last_frame_index];

        l_times_ms = self.create_list_time(list_times_ms, init_time_ms, end_time_ms);
        if len(l_times_ms) < 1:
            raise NameError('[TrackProcess]: len(l_times_ms) < 1 init_frame_index:{} last_frame_index:{}'.format(init_frame_index, last_frame_index))

        # Compute init box:
        pos_init_FNED = self.get_pos_FNED_from_frame_index(init_frame_index);


        # If BM2: Need to smooth with CV to get good initial state
        if self.dynamic_model == 'BM2':

            # Create traker:
            track_ekf = EKF_utils.create_tracker('CV' , self.track_2D.track_id, self.agent_type);
            x_init = track_ekf.create_x_init(0.0, pos_init_FNED[0,0], pos_init_FNED[1,0], 0.0, 0.0);
            track_ekf.set_x_init(x_init, init_time_ms);

            # =========================================
            # Filtering
            # ==========================================
            print('Track_postprocess: Filtering Model: {} track_id: {}'.format('CV', self.track_2D.track_id));

            # Skip first as init already done:
            for current_frame_index in range(init_frame_index+1, last_frame_index+1):

                # Get current time:
                t_current_ms = list_times_ms[current_frame_index];

                # Predict
                track_ekf.kf_predict(t_current_ms);

                # Fuse measurement if there is one
                pix_meas = self.get_pix_meas_from_index(current_frame_index);
                if not (pix_meas is None):
                    track_ekf.kf_fuse(pix_meas, self.cam_model_meas, t_current_ms);

            # =========================================
            # Smoothing
            # ==========================================
            print('Track_postprocess: Smoothing Model: {} track_id: {}'.format('CV', self.get_id()));
            track_ekf.smooth(l_times_ms, post_proces = True);

            # =========================================
            # Generate a trajectory
            # ==========================================

            # Create trajectory from EKF CV
            trajectory = track_ekf.create_trajectory(l_times_ms, self.get_id(), self.agent_type, self.track_2D.color)
            trajectory.complete_missing_psi_rad();

            # Get initial Trajectory point from CV smoothing
            t_current_ms = l_times_ms[0];
            traj_point = trajectory.get_point_at_timestamp(t_current_ms);

            # Reverse initial orientation if needed
            psi_rad = traj_point.psi_rad;
            if reverse_init_BM:
                psi_rad_new = wraptopi(psi_rad + np.pi);
                print('Reverse {} psi_rad: {} psi_rad_new: {}'.format(self.get_id(), psi_rad, psi_rad_new))
                psi_rad = psi_rad_new;

            # Get initial velocity
            v_init = np.sqrt(traj_point.vx*traj_point.vx + traj_point.vy*traj_point.vy);

            # Create BM2 Tracker
            self.track_ekf = EKF_utils.create_tracker('BM2', self.track_2D.track_id, self.agent_type);

            x_init = self.track_ekf.create_x_init(psi_rad, traj_point.x, traj_point.y, traj_point.vx, traj_point.vy);
            self.track_ekf.set_x_init(x_init, t_current_ms);

            # =========================================
            # Filtering
            # ==========================================
            print('Track_postprocess: Filtering Model: {} track_id: {}'.format('BM2', self.track_2D.track_id));

            # Skip first as init already done:
            for current_frame_index in range(init_frame_index+1, last_frame_index+1):

                t_current_ms = list_times_ms[current_frame_index];

                # Predict
                self.track_ekf.kf_predict(t_current_ms);

                # Fuse measurement if there is one
                pix_meas = self.get_pix_meas_from_index(current_frame_index);
                if not (pix_meas is None):
                    self.track_ekf.kf_fuse(pix_meas, self.cam_model_meas, t_current_ms);

            # =========================================
            # Smoothing
            # ==========================================
            print('Track_postprocess: Smoothing Model: {} track_id: {}'.format('BM2', self.get_id()));
            self.track_ekf.smooth(l_times_ms, post_proces = True)

            self.trajectory = self.track_ekf.create_trajectory(l_times_ms, self.get_id(), self.agent_type, self.track_2D.color)

        else:

            # Create new traker:
            self.track_ekf = EKF_utils.create_tracker(self.dynamic_model , self.track_2D.track_id, self.agent_type);
            x_init = self.track_ekf.create_x_init(0.0, pos_init_FNED[0,0], pos_init_FNED[1,0], 0.0, 0.0);
            self.track_ekf.set_x_init(x_init, init_time_ms);

            # =========================================
            # Filtering
            # ==========================================
            print('Track_postprocess: Filtering Model: {} track_id: {}'.format(self.dynamic_model, self.track_2D.track_id));

            # Skip first as init already done:
            for current_frame_index in range(init_frame_index+1, last_frame_index+1):

                # Get current time:
                t_current_ms = list_times_ms[current_frame_index];

                # Predict
                self.track_ekf.kf_predict(t_current_ms);

                # Fuse measurement if there is one
                pix_meas = self.get_pix_meas_from_index(current_frame_index);
                if not (pix_meas is None):
                    self.track_ekf.kf_fuse(pix_meas, self.cam_model_meas, t_current_ms);

            # =========================================
            # Smoothing
            # ==========================================
            print('Track_postprocess: Smoothing track_id: {}'.format(self.get_id()));
            self.track_ekf.smooth(l_times_ms, post_proces = True);

            # =========================================
            # Generate a trajectory
            # ==========================================

            # Create trajectory from EKF CV
            self.trajectory = self.track_ekf.create_trajectory(l_times_ms, self.get_id(), self.agent_type, self.track_2D.color)
            self.trajectory.complete_missing_psi_rad();

##############################################################################
#
# Track merging methods based on Mahalanobis Distance
#
##############################################################################

def compute_merging_cost(tk_process_end, tk_process_start, list_times_ms):
    """Compute the Mahalanobis Distance between two EKF trackers EKF_end and EKF_start:
            - Propagate EKF_end to time_start_ms
            - Compute Innovation covariance between:
                - State of EKF_end time_start_ms: Considered as state
                - State of EKF_start at time_start_ms: Considered as measurement with H = Identity
            - Compute Mahalanobis_distance between state of EKF_end and EKF_start
        This is used to merge tracks based on dynamical model

    Args:
        tk_process_end (TYPE): TrackProcess of track_end
        tk_process_start (TYPE): TrackProcess of track_start
        list_times_ms (TYPE): List of timestamp in ms

    Returns:
        TYPE: Mahalanobis_distance between state of EKF_end and EKF_start
    """
    # Get trajectories
    trajoint_end = tk_process_end.get_trajectory().get_end_trajoint();
    trajoint_start = tk_process_start.get_trajectory().get_start_trajoint();

    label_start = tk_process_start.compute_highest_label();
    label_end = tk_process_end.compute_highest_label();

    # Make sure End Time < Start Time
    if (trajoint_start.time_ms < trajoint_end.time_ms):
        print('[Error]: trajoint_start.time_ms < trajoint_end.time_ms delta_ms: {}'.format(trajoint_start.time_ms < trajoint_end.time_ms));
        return None

    # Get Start / End Time
    start_time_ms = trajoint_start.time_ms;
    end_time_ms = trajoint_end.time_ms;

    # Create list of time_ms between end_time_ms and start_time_ms
    l_times_ms = create_list_time(list_times_ms, end_time_ms, start_time_ms);
    if len(l_times_ms) < 1:
        print('l_times_ms < 1: track_id: {}'.format(self._track_id))

    # Get time init
    t_current_ms = l_times_ms[0];

    # Get EKF tracker
    tk_end_ekf = tk_process_end.get_tracker_EKF();
    tk_start_ekf = tk_process_start.get_tracker_EKF();

    # Get end EKF state
    smooth_state_end = tk_end_ekf.get_tk_smooth(end_time_ms);
    if smooth_state_end is None:
        print('[Error]: smooth_state_end is None');
        return None

    # Get start EKF state
    smooth_state_start = tk_start_ekf.get_tk_smooth(start_time_ms);
    if smooth_state_start is None:
        print('[Error]: smooth_state_start is None');
        return None

    # Get EKF state of EKF_end
    x_init = smooth_state_end.x_state;
    P_init = smooth_state_end.P_cov;

    dynamic_model = tk_end_ekf.get_dynamic_model();
    if dynamic_model != tk_start_ekf.get_dynamic_model():
        raise NameError ('[TrackProcess]: ERROR compute_merging_cost dynamic_model do not match tk_end_ekf: {} tk_start_ekf: {}'.format(tk_end_ekf.get_dynamic_model(), tk_start_ekf.get_dynamic_model()));

    # Init EKF with state of EKF_end: Currently only CV tracker
    track_ekf = EKF_utils.create_tracker(dynamic_model , 0, label_start, P_init = P_init);
    track_ekf.set_x_init(x_init, t_current_ms);

    # Propagate the state from end_times_ms to start_time_ms (skip first time as init already done)
    for t_current_ms in l_times_ms[1:]:

        # Predict
        track_ekf.kf_predict(t_current_ms);

    # Get state of end ekf at start_time_ms
    filt_state_end = track_ekf.get_state_filt_at_time(end_time_ms)

    if filt_state_end is None:
        print('[Error]: filt_state_end is None');
        return None

    # Compute Innovation covariance between state from EKF_end and EKF_start at start_time_ms
    S = filt_state_end.P_cov + smooth_state_start.P_cov;
    x_error = (filt_state_end.x_state - smooth_state_start.x_state);

    # Compute Mahalanobis_distance: https://en.wikipedia.org/wiki/Mahalanobis_distance
    d_sq = x_error.transpose().dot(np.linalg.inv(S).dot(x_error));

    return d_sq;

def get_start_candidate(tk_postprocess_end, tk_postprocess_list, det_zone_FNED, diff_time_ms, list_times_ms):
    """Get TrackProcess candidates for track merge. Return all the track that starts in a diff_time_ms window after tk_postprocess_end ends.

    Args:
        tk_postprocess_end (TYPE): TrackProcess end
        tk_postprocess_list (TYPE): List of all TrackProcess
        det_zone_FNED (TYPE): Detection zone
        diff_time_ms (TYPE): Maximum difference time_ms between end and start of TrackProcess
        list_times_ms (TYPE): List of time_ms

    Returns:
        TYPE: List of TrackProcess candidates
    """
    trajoint_end = tk_postprocess_end.get_trajectory().get_end_trajoint();
    is_inside = det_zone_FNED.in_zone(np.array([[trajoint_end.x], [trajoint_end.y]]));

    candidate_tk_post_start = [];
    if is_inside:
        for tk_postprocess_start in list(tk_postprocess_list):


            trajoint_start = tk_postprocess_start.get_trajectory().get_start_trajoint();
            if (trajoint_start.time_ms > trajoint_end.time_ms) and (trajoint_start.time_ms - trajoint_end.time_ms) < diff_time_ms:

                cost_merge = compute_merging_cost(tk_postprocess_end, tk_postprocess_start, list_times_ms);

                # candidate_tk_post_start.append([tk_postprocess_start,cost_merge]);
                candidate_tk_post_start.append(tk_postprocess_start);

                # print('Tk_post: {} candidate: {} cost: {}'.format(tk_postprocess_end.get_id(), tk_postprocess_2.get_id(), cost_merge))
                # print('End: {} Start: {} Delta: {}'.format(trajoint_end.time_ms, trajoint_start.time_ms,trajoint_start.time_ms - trajoint_end.time_ms))

    return candidate_tk_post_start;


def get_end_candidate(tk_postprocess_start, tk_postprocess_list, det_zone_FNED, diff_time_ms, list_times_ms):
    """Get TrackProcess candidates for track merge. Return all the track that ends in a diff_time_ms window before tk_postprocess_start starts.

    Args:
        tk_postprocess_start (TYPE): TrackProcess start
        tk_postprocess_list (TYPE): List of all TrackProcess
        det_zone_FNED (TYPE): Detection zone
        diff_time_ms (TYPE): Maximum difference time_ms between end and start of TrackProcess
        list_times_ms (TYPE): List of time_ms

    Returns:
        TYPE: List of TrackProcess candidates
    """
    trajoint_start = tk_postprocess_start.get_trajectory().get_start_trajoint();
    is_inside = det_zone_FNED.in_zone(np.array([[trajoint_start.x], [trajoint_start.y]]));

    candidate_tk_post_end = [];
    if is_inside:
        for tk_postprocess_end in list(tk_postprocess_list):


            trajoint_end = tk_postprocess_end.get_trajectory().get_end_trajoint();
            if (trajoint_start.time_ms > trajoint_end.time_ms) and (trajoint_start.time_ms - trajoint_end.time_ms) < diff_time_ms:

                cost_merge = compute_merging_cost(tk_postprocess_end, tk_postprocess_start, list_times_ms);

                # candidate_tk_post_end.append([tk_postprocess_end,cost_merge]);
                candidate_tk_post_end.append(tk_postprocess_end);

                # print('Tk_post: {} candidate: {} cost: {}'.format(tk_postprocess_end.get_id(), tk_postprocess_2.get_id(), cost_merge))
                # print('End: {} Start: {} Delta: {}'.format(trajoint_end.time_ms, trajoint_start.time_ms,trajoint_start.time_ms - trajoint_end.time_ms))

    return candidate_tk_post_end;


