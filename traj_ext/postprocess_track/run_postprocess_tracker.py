# -*- coding: utf-8 -*-

##########################################################################################
#
# POST-PROCESS 3D TRAJECTORY SMOOTHER
#
# Smooth the trajectories with a RTS smoother
#
##########################################################################################

import numpy as np
import time
import cv2
import copy
from scipy.optimize import linear_sum_assignment
import sys
import argparse
import os
import configparser
from shutil import copyfile

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../'));
sys.path.append(ROOT_DIR);

from utils import cfgutil
from utils.mathutil import *

from object_det.mask_rcnn import detect_utils

from tracker.utils import tracker_utils
from tracker import cameramodel as cm

from postprocess_track import trajutil
from postprocess_track import track_postprocess

from box3D_fitting import Box3D_utils
from camera_calib import calib_utils

# Initial Parameters

# Create a default cfg file which holds default values for the path
def create_default_cfg():

    config = configparser.ConfigParser();

    config['INPUT_PATH'] = \
                        {'CAMERA_CFG_STREET_PATH': '',\
                         'CAMERA_CFG_SAT_PATH':  '',\
                         'CAMERA_IMG_SAT_PATH':  '',\
                         'DET_ZONE_F_PATH':  '',\
                         'IMAGE_DATA_DIR':  '',\
                         'DET_DATA_DIR':  '',\
                         'BOX3D_DATA_DIR':  '',\
                         'DET_ASSOCIATION_DATA_DIR':  ''}

    config['OUTPUT_PATH'] = \
                        {'tracker_output_dir': ''}

    config['OPTIONS'] = \
                        {'DYNAMIC_MODEL': 'CV', \
                         'PLOT_FILTER': 0,\
                         'PLOT_SMOOTHER': 1,\
                         'SAVE_CSV': 1,\
                         'PLOT_MEAS': 0,\
                         'DELTA_MS': '',
                         'BOX_OVERLAP_THRESHOLD' : 0.4,
                         'MODE_2D_3D': 'box3d'};

    # Header of the cfg file
    text = '\
##########################################################################################\n\
#\n\
# POST-PROCESS 3D TRAJECTORY SMOOTHER\n\
#\n\
# Please modify this config file according to your configuration.\n\
# Path must be ABSOLUTE PATH\n\
##########################################################################################\n\n'

    # Write the cfg file
    with open('TRACKER_POSTPROCESS_CFG.ini', 'w') as configfile:
        configfile.write(text);
        config.write(configfile)

def get_tk_postprocess_in_list(tk_postprocess_list, id):

    for tk_postprocess in tk_postprocess_list:
        if id == tk_postprocess.get_id():
            return tk_postprocess;

    return None;

def main():

# Print instructions
    print("############################################################")
    print("Post-Process Trajectory Smoother")
    print("############################################################\n")

    # ##########################################################
    # # Parse Arguments
    # ##########################################################
    argparser = argparse.ArgumentParser(
        description='Post-Process Trajectory Smoother')
    argparser.add_argument(
        '-i', '--init',
        action='store_true',
        help='Generate config files to fill in order to run the smoother: TRACKER_POSTPROCESS_CFG.ini')
    argparser.add_argument(
        '-cfg', '--config',
        default='TRACKER_POSTPROCESS_CFG.ini',
        help='Path to the config file')
    args = argparser.parse_args();

    # In init mode
    if args.init:

        # Create default config file
        create_default_cfg();

        print('Please fill the config files and restart the program:\n-TRACKER_POSTPROCESS_CFG.ini')

        return;

    # ##########################################################
    # # Read config file:
    # ##########################################################
    config = cfgutil.read_cfg(args.config);


    # Create output dorectory to save filtered image:
    tracker_img_dir = os.path.join(config['OUTPUT_PATH']['tracker_output_dir'], 'img');
    os.makedirs(tracker_img_dir, exist_ok=True)
    tracker_csv_dir = os.path.join(config['OUTPUT_PATH']['tracker_output_dir'], 'csv');
    os.makedirs(tracker_csv_dir, exist_ok=True)

    # Save the cfg file with the output:
    try:
        cfg_save_path = args.config.split('/')[-1];
        cfg_save_path = os.path.join(config['OUTPUT_PATH']['tracker_output_dir'], cfg_save_path);
        copyfile(args.config, cfg_save_path)
    except Exception as e:
        print('[Error]: Error saving config file in output folder:\n')
        print('{}'.format(e))
        return;

    # Options for the outputs
    SAVE_CSV = False;
    if bool(float(config['OPTIONS']['SAVE_CSV'])):
      SAVE_CSV = True;


    ##########################################################
    # Camera Parameters
    ##########################################################

    # Camera model street
    cam_model_1 = calib_utils.read_camera_calibration(config['INPUT_PATH']['CAMERA_CFG_STREET_PATH']);

    # Camera sat
    cam_model_2 = calib_utils.read_camera_calibration(config['INPUT_PATH']['CAMERA_CFG_SAT_PATH']);

    # ######################################################
    # Initialize parameters :
    # ######################################################

    plot_smoother = bool(float(config['OPTIONS']['PLOT_SMOOTHER']))
    plot_filter = bool(float(config['OPTIONS']['PLOT_FILTER']))
    plot_meas = bool(float(config['OPTIONS']['PLOT_MEAS']))

    overlap_threshold = float(config['OPTIONS']['BOX_OVERLAP_THRESHOLD'])

    kf_tracker_list = []
    list_times_ms =[]
    list_img = [];

    # Open img folder
    list_img_file = os.listdir(config['INPUT_PATH']['IMAGE_DATA_DIR']);
    list_img_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))));
    list_img_file = list_img_file

    # Get timestamp from cfg file if specified
    t_current_ms = 0; # TO resolve
    timestamp_img_bool = True;

    if not config['OPTIONS']['DELTA_MS'] == '':
        delta_ms = int(config['OPTIONS']['DELTA_MS']);
        timestamp_img_bool = False;
        t_last_ms = None;

    # Get timestamp from cfg image name otherwise
    else:
        t_last_ms = None;
        timestamp_img_bool = True;

    # Get satellite image
    im_sat_2 = cv2.imread(config['INPUT_PATH']['CAMERA_IMG_SAT_PATH']);
    if im_sat_2 is None:
        print('\n[Error]: camera_img_sat_path is not valid: {}'.format(config['INPUT_PATH']['CAMERA_IMG_SAT_PATH']));
        return;

    # Get mode of 2D to 3D conversion:
    mode_2D_3D = config['OPTIONS']['MODE_2D_3D'];

    ##########################################################
    # Detection zone
    ##########################################################

    # Detection Zone YML file:
    pt_det_zone_FNED = None;
    if config['INPUT_PATH']['DET_ZONE_F_PATH'] != '':
        fs_read = cv2.FileStorage(config['INPUT_PATH']['DET_ZONE_F_PATH'], cv2.FILE_STORAGE_READ)
        pt_det_zone_FNED = fs_read.getNode('model_points_FNED').mat()

    # ######################################################
    # Go through detection from image to image
    # ######################################################

    # List of trajectories:
    tk_postprocess_list = [];
    list_times_ms = [];

    # Method filtering and smoothing with running window:
    # - list_times: Running window of size window_width
    #      - list_times[-1]: last time is filter time
    #      - list_times[0]:  First time is smoothing time


    if len(list_img_file) > 0:
        name_prefix = list_img_file[0].split('.')[0];

    init_flag = True;
    for current_index, current_img_file in enumerate(list_img_file):

        # ######################################################
        # Getting detetions
        # ######################################################

        img_file = list_img_file[current_index]

        # Open box 3D and track CSV:
        box3D_csv_name = img_file.split('.')[0] + '_3Dbox.csv';
        track_csv_name = img_file.split('.')[0] + '_detassociation.csv';
        det_csv_name = img_file.split('.')[0] + '_det.csv';

        try:
            data_det_list = detect_utils.read_dict_csv(os.path.join(config['INPUT_PATH']['DET_DATA_DIR'], det_csv_name));
            data_box3D_list = detect_utils.read_dict_csv(os.path.join(config['INPUT_PATH']['BOX3D_DATA_DIR'], box3D_csv_name));
            data_track_list = detect_utils.read_dict_csv(os.path.join(config['INPUT_PATH']['DET_ASSOCIATION_DATA_DIR'], track_csv_name));
        except Exception as e:
            print("ERROR: Could not open csv: {}".format(e));
            data_box3D_list = [];
            data_track_list = [];

        # ######################################################
        # Remove bad detections
        # ######################################################

        # Get trackers of detection zone
        data_box3D_list = tracker_utils.detect_tracking_zone(data_box3D_list, pt_det_zone_FNED)

        # Remove 3D box detection that are bellow the overlap threshold: 3D box and detection mask overlap bellow threshold
        data_box3D_list = tracker_utils.remove_3Dbox_threshold(data_box3D_list, overlap_threshold);

        # ######################################################
        # Manage time
        # ######################################################

        # Get time from file name
        if timestamp_img_bool:
            t_current_ms, t_last_ms, delta_ms = tracker_utils.get_time(img_file, t_current_ms)

        # Time from delta_s from cfg: start from t = 0 and increment by delta_s
        else:
            # Start from t = 0
            if init_flag:
                t_current_ms = 0
                init_flag = False;

            # Increment time with delta_s
            else:
                t_current_ms = list_times_ms[-1] + delta_ms;

        list_times_ms.append(t_current_ms);

        print('\n[INFO] Reading step: ', current_index)
        print('[INFO] Reading time ms: ', t_current_ms)

        # ######################################################
        # Create tracks object with associated measurements
        # ######################################################

        label_list = ['car','bus','truck'];
        for data_track in data_track_list:

            track_id = data_track['track_id'];
            det_id = data_track['det_id'];

            box3D = None;
            for data_box3D in data_box3D_list:
                if data_box3D['det_id'] == det_id:
                    box3D = data_box3D['box_3D']

            label = None;
            for data_det in data_det_list:
                if data_det['det_id'] == det_id:
                    label = data_det['label'];

            if label in label_list and not (box3D is None):

                tk_postprocess = get_tk_postprocess_in_list(tk_postprocess_list, track_id);

                # If trajectory is not created yet, create it
                if tk_postprocess is None:
                    tk_postprocess = track_postprocess.Track_postprocess(track_id, cam_model_1, config['OPTIONS']['DYNAMIC_MODEL']);
                    tk_postprocess_list.append(tk_postprocess);


                # ######################################################
                # Use the center of bottom of 3D box
                # ######################################################

                if mode_2D_3D == 'box3d':

                    # Add point to the corresponding trajetcory
                    tk_postprocess.push_3Dbox_meas(t_current_ms, box3D, label);

                # ######################################################
                # Use Center of 2D detected bounding box
                # ######################################################

                elif mode_2D_3D == 'center_2d':

                    det_roi = None;
                    for data_det in data_det_list:
                        if data_det['det_id'] == det_id:
                            det_roi = data_det['roi'];

                    if not (det_roi is None):

                        # Get ROI coordinates
                        x_1 = int(det_roi[1]);
                        y_1 = int(det_roi[0]);
                        x_2 = int(det_roi[3]);
                        y_2 = int(det_roi[2]);

                        # Compute rough 3D position of the object:
                        # Re-project the center of the ROI on the ground

                        # Option 1: Project center of the 2D bounding box
                        pt_image_x = (x_1 + x_2)/2
                        pt_image_y = (y_1 + y_2)/2

                        pix_meas = np.array([int(pt_image_x), int(pt_image_y)]);
                        pix_meas.shape = (2,1);

                        # Add point to the corresponding trajetcory
                        tk_postprocess.push_pix_meas(t_current_ms, pix_meas, label);

                # #############################################################
                # Use Center of the bottom side of the 2D detected bounding box
                # #############################################################

                elif mode_2D_3D == 'center_bottom_2d':

                    det_roi = None;
                    for data_det in data_det_list:
                        if data_det['det_id'] == det_id:
                            det_roi = data_det['roi'];

                    if not (det_roi is None):

                        # Get ROI coordinates
                        x_1 = int(det_roi[1]);
                        y_1 = int(det_roi[0]);
                        x_2 = int(det_roi[3]);
                        y_2 = int(det_roi[2]);

                        # Compute rough 3D position of the object:
                        # Re-project the center of the ROI on the ground


                        # Option 2: Project mid point of the bottom edgethe 2D bounding box
                        pt_image_x = (x_1 + x_2)/2;
                        pt_image_y = y_2;


                        pix_meas = np.array([int(pt_image_x), int(pt_image_y)]);
                        pix_meas.shape = (2,1);

                        # Add point to the corresponding trajetcory
                        tk_postprocess.push_pix_meas(t_current_ms, pix_meas, label);

                else:
                    raise Exception('[Error] Mode mode_2D_3D not recognized: {} \nIt should be box3d / center_2d / center_bottom_2d'.format(mode_2D_3D));

    # ######################################################
    # Post-Process each tracks
    # ######################################################

    # Filter and smooth all trajectories
    for tk_postprocess in tk_postprocess_list:

        tk_postprocess.process_traj(list_times_ms);

    print('[INFO]: Number of track: {}'.format(len(tk_postprocess_list)));

    # ==========================================
    #         Clean trajectories
    # ==========================================

    # remove tracks that are too short
    for tk_postprocess in list(tk_postprocess_list):

        if not(tk_postprocess.get_trajectory() is None):
            if tk_postprocess.get_trajectory().get_length_ms() < 1000:
                print('Removing traj: {} traj length: {} ms'.format(tk_postprocess.get_id(), tk_postprocess.get_trajectory().get_length_ms()));
                tk_postprocess_list.remove(tk_postprocess);

    print('[INFO]: Number of track: {}'.format(len(tk_postprocess_list)));

    # ==========================================
    #         Write trajectory csv
    # ==========================================
    trajectory_list = [];
    for tk_postprocess in tk_postprocess_list:

        if not(tk_postprocess.get_trajectory() is None):
            trajectory_list.append(tk_postprocess.get_trajectory());

    csv_path = config['OUTPUT_PATH']['tracker_output_dir'] + '/csv';
    trajutil.write_trajectory_csv(csv_path, name_prefix, trajectory_list, list_times_ms);

    # ==========================================
    #         Plot Filter and Smoother
    # ==========================================

    # TEMPORARY each tracks
    kf_tracker_list = [];
    for tk_postprocess in tk_postprocess_list:

        kf_track = tk_postprocess.get_tracker_EKF();
        if not(kf_track is None):
            kf_tracker_list.append(kf_track);

    for current_index, current_img_file in enumerate(list_img_file):

        t_smooth_ms = list_times_ms[current_index];

        img_smooth = current_img_file;

        # Getting current street image
        img_street = cv2.imread(os.path.join(config['INPUT_PATH']['IMAGE_DATA_DIR'], img_smooth));
        print('img_smooth: {}'.format(img_smooth))

        # Copy to get a clean image to show detection for each frame
        img_sat = copy.copy(im_sat_2);

        if not (pt_det_zone_FNED is None):
            # Draw tracking zone:
            Box3D_utils.draw_det_zone(img_sat, cam_model_2, pt_det_zone_FNED, color=(0,0,255), thickness=2);
            Box3D_utils.draw_det_zone(img_street, cam_model_1, pt_det_zone_FNED, color=(0,0,255), thickness=2);

            # mask_tracking_zone_sat = Box3D_utils.create_mask((img_sat.shape[0], img_sat.shape[1]), cam_model_2, pt_det_zone_FNED);
            # Box3D_utils.draw_mask(img_sat, mask_tracking_zone_sat, (0,0,255));

            # mask_tracking_zone_street = Box3D_utils.create_mask((img_street.shape[0], img_street.shape[1]), cam_model_1, pt_det_zone_FNED);
            # Box3D_utils.draw_mask(img_street, mask_tracking_zone_street, (0,0,255));

        img_sat, img_street = tracker_utils.construct_plot_trackers(t_smooth_ms, kf_tracker_list, 50, img_street, cam_model_1, img_sat, cam_model_2, plot_smoother, plot_filter, plot_meas);

        tracker_utils.show_trackers(img_street, img_sat, img_smooth, config['OUTPUT_PATH']['tracker_output_dir']);

        # Normal Mode: Press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;

        print('[Info]: Plotting step: {}'.format(current_index))
        # time.sleep(0.1);

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')