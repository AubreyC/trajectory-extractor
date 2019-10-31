# -*- coding: utf-8 -*-

########################################################################################
#
# Implementation of utils function for filtering and smoothing
#
########################################################################################

import numpy as np
import cv2
import csv
import sys
import copy
import os
import time

#Import the tracker EKF model:
from traj_ext.tracker import EKF_CVCYR
from traj_ext.tracker import EKF_CV
from traj_ext.tracker import EKF_BM2

def create_tracker(dynamic_model, track_id, label, P_init = None, Q = None, R = None):
    """Create a EKF tracker with specific dynamic model

    Args:
        dynamic_model (TYPE): Dynamic Model
        track_id (TYPE): Track id
        label (TYPE): Agent Type / Label
        P_init (None, optional): Initial state covariance
        Q (None, optional): Process noise
        R (None, optional): Measurement noise

    Returns:
        EKF_tracker object

    Raises:
        ValueError: Unknown Dynamic model
    """

    # Constant Velocity Model:
    if dynamic_model == "CV":

        P_init_def, Q_def, R_def = EKF_CV.EKF_CV_track.get_default_param();
        if P_init is None:
            P_init = P_init_def;
        if Q is None:
            Q = Q_def;
        if R is None:
            R = R_def;


        # Create new traker:
        tracker = EKF_CV.EKF_CV_track(Q, R, P_init, track_id, label);

    # Constant Velocity Constant Yaw Rate Model:
    elif dynamic_model == "CVCYR":

        P_init_def, Q_def, R_def = EKF_CVCYR.EKF_CVCYR_track.get_default_param();
        if P_init is None:
            P_init = P_init_def;
        if Q is None:
            Q = Q_def;
        if R is None:
            R = R_def;

        # Create new traker:
        tracker = EKF_CVCYR.EKF_CVCYR_track( Q, R, P_init, track_id, label);

    # Bicycle model
    elif dynamic_model == "BM2":

        P_init_def, Q_def, R_def = EKF_BM2.EKF_BM2_track.get_default_param();
        if P_init is None:
            P_init = P_init_def;
        if Q is None:
            Q = Q_def;
        if R is None:
            R = R_def;

        # Create new traker:
        tracker = EKF_BM2.EKF_BM2_track( Q, R, P_init, track_id, label);

    # Raise Error if the model is not recognized
    else:
        raise ValueError('Dynamic Model not handled: {}'.format(dynamic_model));


    return tracker;

def concatenate_images(img_1, img_2):
    """Concatenate images into one

    Args:
        img_1 (TYPE): Description
        img_2 (TYPE): Description

    Returns:
        TYPE: Description

    """
    res_4 = None;
    if not (img_1 is None):
        # Resize Camera and Satellite Image:
        res_1 = cv2.resize(img_2, None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
        res_2 = cv2.resize(img_1, None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)

        #Concatenate Camera and Satellite view on single image
        h_1 = res_1.shape[0];
        w_1 = res_1.shape[1];
        h_2 = res_2.shape[0];
        w_2 = res_2.shape[1];
        scale = float(h_1)/float(h_2);

        h_2 = h_1;
        w_2 = int(w_2*scale)
        dim = (w_2, h_2);
        res_3 = cv2.resize(res_2, dim, interpolation = cv2.INTER_CUBIC)

        res_4 = np.concatenate((res_1, res_3), axis=1)

    return res_4;