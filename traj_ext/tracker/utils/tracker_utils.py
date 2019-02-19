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
from tracker import EKF_CVCYR
from tracker import EKF_CV
from tracker import EKF_BM
from tracker import EKF_BM2

from tracker import cameramodel as cm
from box3D_fitting import Box3D_utils
from object_det.mask_rcnn import detect_utils


def get_time(current_img_file, t_last_s):

    """Gets time of frame from timestamp or using time increments

    Args:
        current_img_file (string): Description
        t_current (float): Description
        delta_s (float): Description
        timestamp_img_bool (boolean): Description

    Returns:
        float: time corresponding to the current_img_file
    """
	# Get timestamp from image name:

    # From image name if DELTA_MS not defined in cfg file:
    t_current = current_img_file.split('_')[-1];
    t_current = t_current.split('.')[0];
    t_current = int(t_current);
    t_current = np.float64(t_current)/1e3;
    # TO DO: Should compute time step from the actual frame timestamp
    if not(t_last_s is None):
        delta_s = t_current - t_last_s;
    else:
        delta_s = 0

    t_last_s = t_current;

    return t_current, t_last_s, delta_s

def get_filter_param(dynamic_model):
    """ Initialize parameters for each dynamic model
		Return the Filter Matrix parameters for specific dynamic model

    Args:
        dynamic_model (string)

    Returns:
        P_init (matrix): Initial covariance
        Q (matrix): Process noise
        R (matrix): Measurement noise

    Raises:
        ValueError: Description
    """
    Q = None;
    R = None;
    P_init = None;

    # Constant Velocity Model:
    if dynamic_model == "CV":
        # Process noise
        Q = np.matrix([[ 0,  0,    0,    0],\
                       [ 0,  0,    0,    0],\
                       [ 0,  0,    0.5,    0],\
                       [ 0,  0,    0,    0.5]], np.float64);

        # Measurement noise: Pixel Noise
        R = np.matrix([[ 80,  0],\
                       [  0, 80]], np.float64);


        # Init Covariance
        P_init = np.matrix([[ 0.1,  0,    0,    0],\
                            [ 0,  0.1,    0,    0],\
                            [ 0,  0,    10,    0],\
                            [ 0,  0,    0,    10]], np.float64);



    # Constant Velocity Constant Yaw Rate Model:
    elif dynamic_model == "CVCYR":
        # Process noise
        Q = np.matrix([[ 0.1,  0,   0,    0,     0],\
                       [ 0,  0.1,   0,    0,     0],\
                       [ 0,  0,     1,    0,     0],\
                       [ 0,  0,     0,    0,     0],\
                       [ 0,  0,     0,    0,     0.2]], np.float64);

        # Measurement noise: Pixel Noise
        R = np.matrix([[ 120,  0],\
                       [  0, 120]], np.float64);

        # Init Covariance
        P_init = np.matrix([[   0.1,     0,    0,    0,    0],\
                            [   0,     0.1,    0,    0,    0],\
                            [   0,     0,    2,    0,    0],\
                            [   0,     0,    0,    2,    0],\
                            [   0,     0,    0,    0,    0.01]], np.float64);

    # Constant Velocity Constant Yaw Rate Model:
    elif dynamic_model == "BM":

        # Process noise
        Q = np.matrix([[ 0.1,  0,     0,    0,     0],\
                       [ 0,  0.1,     0,    0,     0],\
                       [ 0,  0,     0.5,    0,     0],\
                       [ 0,  0,     0,      0,     0],\
                       [ 0,  0,     0,    0,     0.1]], np.float64);

        # Measurement noise: Pixel Noise
        R = np.matrix([[ 100,  0],\
                       [  0, 100]], np.float64);

        # Init Covariance
        P_init = np.matrix([[   0.1,     0,    0,    0,    0],\
                            [   0,     0.1,    0,    0,    0],\
                            [   0,     0,    0.1,    0,    0],\
                            [   0,     0,    0,    1,    0],\
                            [   0,     0,    0,    0,     0.01]], np.float64);

        # Constant Velocity Constant Yaw Rate Model:
    elif dynamic_model == "BM2":

        # Process noise
        Q = np.matrix([[ 0.1,  0,     0,    0,     0],\
                       [ 0,  0.1,     0,    0,     0],\
                       [ 0,  0,     0.5,    0,     0],\
                       [ 0,  0,     0,      0,     0],\
                       [ 0,  0,     0,    0,     0.1]], np.float64);

        # Measurement noise: Pixel Noise
        R = np.matrix([[ 80,  0],\
                       [  0, 80]], np.float64);

        # Init Covariance
        P_init = np.matrix([[   0.1,     0,    0,    0,    0],\
                            [   0,     0.1,    0,    0,    0],\
                            [   0,     0,    1,    0,    0],\
                            [   0,     0,    0,    20,    0],\
                            [   0,     0,    0,    0,     0.01]], np.float64);


    # Raise Error if the model is not recognized
    else:
        raise ValueError('Dynamic Model not handled: {}'.format(dynamic_model));


    return P_init, Q, R;

def detect_tracking_zone(data_box3D_list, pt_det_zone_FNED):
    """Limit the filtering to the detection within a zone by keeping only objects present within the zone

    Args:
        data_box3D_list (list)
        pt_det_zone_FNED (bool)

    Returns:
        list
    """

    if pt_det_zone_FNED is not None:

        for data_box3D in list(data_box3D_list):
            box3D = data_box3D['box_3D']
            pt = (box3D[1], box3D[2])

            if pt_det_zone_FNED is not None:
                # Detection Zone from yml
                contour = pt_det_zone_FNED.astype(int); (pt_det_zone_FNED)
                contour = contour[:,0:2]

                res = cv2.pointPolygonTest(contour, pt, False)
                if res < 1:
                    data_box3D_list.remove(data_box3D)

    return data_box3D_list

def remove_3Dbox_threshold(data_box3D_list, threshold):
    """Remove 3D box that are under the overlap trheshold
       Overlap between 3D box and mask

    Args:
        data_box3D_list (list): Description
        threshold (list): Description

    Returns:
        TYPE: Description
    """

    # Clean not accurate box fitting
    for data_box3D in list(data_box3D_list):
        if data_box3D['percent_overlap'] < threshold:
            data_box3D_list.remove(data_box3D);

    return data_box3D_list;


def create_tracker(dynamic_model, box3D,  track_id, t_current, label):
    """ Create a Filter tracker with specific dynamic model

    Args:
        dynamic_model (string)
        box3D (list)
        track_id (int)

    Returns:
        EKF_tracker object

    Raises:
        ValueError: Description
    """
    P_init, Q, R = get_filter_param(dynamic_model);

    # Constant Velocity Model:
    if dynamic_model == "CV":

        # Create init state from 3D box:

        # Create new traker:
        tracker = EKF_CV.EKF_CV_track(Q, R, None, P_init, track_id, t_current, label);
        tracker.set_x_init(box3D);
        tracker.set_phi_rad(np.deg2rad(box3D[0]), t_current)


    # Constant Velocity Constant Yaw Rate Model:
    elif dynamic_model == "CVCYR":

        # Create new traker:
        tracker = EKF_CVCYR.EKF_CVCYR_track( Q, R, None, P_init, track_id, t_current, label);
        tracker.set_x_init(box3D);

    # Constant Velocity Constant Yaw Rate Model:
    elif dynamic_model == "BM":

        # Create new traker:
        tracker = EKF_BM.EKF_BM_track( Q, R, None, P_init, track_id, t_current, label);
        tracker.set_x_init(box3D);

    # Constant Velocity Constant Yaw Rate Model:
    elif dynamic_model == "BM2":

        # Create new traker:
        tracker = EKF_BM2.EKF_BM2_track( Q, R, None, P_init, track_id, t_current, label);
        tracker.set_x_init(box3D);

    # Raise Error if the model is not recognized
    else:
        raise ValueError('Dynamic Model not handled: {}'.format(dynamic_model));


    return tracker;

def get_3Dbox_by_detid(det_id, data_box3D_list):

    box3D = None;
    for data_box3D in data_box3D_list:

        if data_box3D['det_id'] == det_id:
            box3D = data_box3D['box_3D'];

    return box3D;

def get_trackid_by_detid(det_id, data_track_list):

    track_id = None;
    for data_track in data_track_list:
        if data_track['det_id'] == det_id:
            track_id = data_track['track_id'];

    return track_id;

def get_label_by_detid(det_id, data_det_list):

    label = None;

    for data_det in data_det_list:
        if data_det['det_id'] == det_id:
            label = data_det['label'];

    return label

def get_pixprojection_from_box(box3D, cam_model_1):

        # Project on Image
        pt_F = np.array(box3D[1:4]);
        pt_F.shape = (3,1)

        (pt_img, jacobian) = cv2.projectPoints(pt_F.transpose(), cam_model_1.rot_CF_F, cam_model_1.trans_CF_F, cam_model_1.cam_matrix, cam_model_1.dist_coeffs)
        pt_img_tulpe = (int(pt_img[0][0][0]), int(pt_img[0][0][1]));

        pix_meas = np.array([int(pt_img[0][0][0]), int(pt_img[0][0][1])]);
        pix_meas.shape = (2,1);

        return pix_meas

def construct_plot_trackers(t_current, kf_tracker_list, window_width, img_street, cam_model_street, img_sat, cam_model_sat, plot_smoother = True, plot_filter = False, plot_meas = False):
    """ Initialize plotting variables

    Args:
        kf_tracker_list (list)
        t_current (float)
        config (list)
        im_sat_2 (matrix)
        cam_model_sat (CameraModel object)
        cam_model_street (CameraModel object)

    Returns:
        satellite image, street view image, variable to be stored in cvs
    """
    # Open Corresponding image:

    for tk in list(kf_tracker_list):

        # =========================================
        #               Plot filtering
        # ==========================================

        # Get the filterd 3D Box model from the tracker
        box3D_filt = tk.get_3Dbox_filt(t_current);

        if not(box3D_filt is None):

            if plot_filter:

                # Create the corner points for the 3D Box measured
                list_pt_F_filt = Box3D_utils.create_3Dbox(box3D_filt);
                xy, vxy, phi_rad = tk.get_filt_pos(t_current);

                if not (img_sat is None):

                # for state in (reversed(tk.x_list[window_width:])):
                #     # Reproject on the satellite image
                #     pt_pix = cam_model_sat.project_points(np.array([(state[0,0], state[1,0], 0.0)]));
                #     pt_pix = (int(pt_pix[0]),int(pt_pix[1]));
                #     cv2.circle(img_sat, pt_pix,3, tk.color, -1);

                    img_sat = Box3D_utils.draw_3Dbox(img_sat, cam_model_sat, list_pt_F_filt, (0, 255,0)); # GREEN

                if not (img_street is None):

                    # Plot 3D box on view image
                    img_street = Box3D_utils.draw_3Dbox(img_street, cam_model_street, list_pt_F_filt, (0, 255,0)); # GREEN

                    pt_pix = cam_model_street.project_points(np.array([(xy[0], xy[1], 0.0)]));
                    pt_pix = (int(pt_pix[0]),int(pt_pix[1]));
                    cv2.circle(img_street, pt_pix, 6, tk.color, -1)

                    # # Add annotations to the track:
                    # text = "id: %i" % (tk.track_id);
                    # img_street = cv2.putText(img_street, text, pt_pix, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)

        # ==========================================
        #               Plot smoothing
        # ==========================================

        # if plot_smoother and tk.is_active(t_current):
        if plot_smoother:

            # Plot past traj points:
            if not (img_sat is None):

                for traj_point in reversed(tk.get_traj_smooth(t_current, window_width)):

                    # Reproject on the satellite image
                    pt_pix = cam_model_sat.project_points(np.array([(traj_point.x, traj_point.y, 0.0)]));
                    pt_pix = (int(pt_pix[0]),int(pt_pix[1]));
                    cv2.circle(img_sat, pt_pix, 6, tk.color, -1);

            # Plot Smoothed Box:
            box3D_smooth = tk.get_3Dbox_smooth(t_current);
            if not(box3D_smooth is None) :

                # Get point:
                list_pt_S_smoothed = Box3D_utils.create_3Dbox(box3D_smooth);

                # Draw box on images
                img_sat = Box3D_utils.draw_3Dbox(img_sat, cam_model_sat, list_pt_S_smoothed, (255, 0, 0)); #BLUE
                img_street = Box3D_utils.draw_3Dbox(img_street, cam_model_street, list_pt_S_smoothed,(255, 0, 0) ); # BLUE


            # Plot smooth data:
            xy, vxy, phi_rad = tk.get_processed_parameters_smoothed(t_current);
            if not (vxy is None):

                if not (img_sat is None):

                    # Plot on Image Sat
                    pt_pix = cam_model_sat.project_points(np.array(([xy[0], xy[1], 0.0])));
                    pt_pix = (int(pt_pix[0]),int(pt_pix[1]));
                    cv2.circle(img_sat, pt_pix, 8, tk.color, 3)


                    # Add annotations to the track:
                    text = "id: %i" % (tk.track_id);
                    img_sat = cv2.putText(img_sat, text, pt_pix, cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)

                    # # Add covariance ellipse:
                    # pos_F = np.asarray(smooth_data.x_state[0:2,0]).reshape(-1);
                    # pos_F = np.append(pos_F, 0);
                    # H = cam_model_sat.compute_meas_H(pos_F);
                    # H = np.append(H, np.zeros((2,1)), axis=1);

                    # mat_cov = H.dot(smooth_data.P_cov.dot(H.transpose()));
                    # cm.compute_ellipse(img_sat, 2.4477, pt_pix, mat_cov);

                if not (img_street is None):

                    # Plot on Street Image
                    pt_pix = cam_model_street.project_points(np.array([(xy[0], xy[1], 0.0)]));
                    pt_pix = (int(pt_pix[0]),int(pt_pix[1]));
                    cv2.circle(img_street, pt_pix, 8, tk.color, 3)

                    # Add annotations to the track:
                    text = "id: %i" % (tk.track_id);
                    img_street = cv2.putText(img_street, text, pt_pix, cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)

        # ==========================================
        #               Plot Measurement
        # ==========================================

        # Show 3D Box filt on satellite image
        if plot_meas:

            box3D_meas = tk.get_3Dbox_meas(t_current);
            if not(box3D_meas is None):

                list_pt_F = Box3D_utils.create_3Dbox(box3D_meas);

                if not (img_sat is None):

                    # Plot on Imge Sat
                    img_sat = Box3D_utils.draw_3Dbox(img_sat, cam_model_sat, list_pt_F, (0, 0, 255)); # RED

                    pt_pix = cam_model_sat.project_points(np.array([(box3D_meas[1], box3D_meas[2], 0.0)]));
                    pt_pix = (int(pt_pix[0]),int(pt_pix[1]));
                    cv2.circle(img_sat, pt_pix, 4 , (0, 255,255), -1);

                if not (img_street is None):

                    # Plot on Imge Sat
                    img_street = Box3D_utils.draw_3Dbox(img_street, cam_model_street, list_pt_F, (0, 0, 255)); # RED

                    pt_pix = cam_model_street.project_points(np.array([(box3D_meas[1], box3D_meas[2], 0.0)]));
                    pt_pix = (int(pt_pix[0]),int(pt_pix[1]));
                    cv2.circle(img_street, pt_pix, 4 , (0, 255,255), -1);


    return img_sat, img_street

def show_trackers(img_street, img_sat, current_img_file, output_dir, no_background = False, img_street_nb = None, img_sat_nb = None):
    """Plot trackers on street and satellite images

    Args:
        img_street (array)
        img_sat (float)
        current_img_file (string)
        config (list)
    """

    # cv2.imshow('img_street', img_street)
    # cv2.imshow('img_sat', img_sat)
    # return;

    if not (img_street is None):
        # Resize Camera and Satellite Image:
        res_1 = cv2.resize(img_sat, None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
        res_2 = cv2.resize(img_street, None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)

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

        if no_background:

            print('img_street_nb shape: {}'.format(img_street_nb.shape))
            print('img_sat_nb shape: {}'.format(img_sat_nb.shape))

            res_1 = cv2.resize(img_sat_nb, None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
            res_2 = cv2.resize(img_street_nb, None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
            #Concatenate Camera and Satellite view on single image
            h_1 = res_1.shape[0];
            w_1 = res_1.shape[1];
            h_2 = res_2.shape[0];
            w_2 = res_2.shape[1];
            scale = float(h_1)/float(h_2);

            h_2 = h_1;
            w_2 = int(w_2*scale)
            dim = (w_2, h_2);
            res_3 = cv2.resize(res_2,dim, interpolation = cv2.INTER_CUBIC)

            res_4_bis = np.concatenate((res_1, res_3), axis=1)

            # Resizing
            res_4 = cv2.resize(res_4, None, fx=0.8, fy=0.8,  interpolation = cv2.INTER_CUBIC)
            res_4_bis = cv2.resize(res_4_bis, None, fx=0.8, fy=0.8, interpolation = cv2.INTER_CUBIC)

            # Concatenate
            res_4 = np.concatenate((res_4, res_4_bis), axis=0)

        cv2.imshow('View', res_4)

        img_track = current_img_file.split('.')[0] + "_filter.png"
        path_img_track = os.path.join(output_dir + '/img', img_track)

        cv2.imwrite(path_img_track, res_4 );