#################################################################################
#
# Trajectory test
#
#################################################################################

import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
import sys
import cv2
import time

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../../'));
sys.path.append(ROOT_DIR);

from utils import cfgutil
from utils.mathutil import *

from postprocess_track import trajutil
from postprocess_track import trajectory

from tracker import cameramodel as cm
from camera_calib import calib_utils

def compute_precision(traj_list, traj_list_truth, display_hist=False, ):

    print('[INFO]: Trajectories {}'.format(len(traj_list)))

    for traj in traj_list_truth:
        print('[INFO]: Traj {} lenght: {} start_t: {} end_t: {}'.format(traj.get_id(), traj.get_length_ms(), traj.get_traj()[0].time_ms, traj.get_traj()[-1].time_ms));


    total_legnth_ms = 0;
    for traj in list(traj_list):
        if traj.get_length_ms() < 1000:
            print('Removing traj: {} traj length: {} ms'.format(traj.get_id(), traj.get_length_ms()));
            traj_list.remove(traj);
        else:
            total_legnth_ms = total_legnth_ms + traj.get_length_ms();

    result_error_xy = [];
    result_error_vxvy = [];
    result_error_psi_rad = [];


    result_traj_truth_list = [];
    result_traj_list = [];

    for traj_1 in traj_list:

        # Test to find the closest match:
        index, traj_min = trajutil.find_closest_traj(traj_1, traj_list_truth, traj_included = True);
        result_traj_truth_list.append(traj_min);
        result_traj_list.append(traj_1);

        if not (traj_min is None):
            error_xy, error_vxvy, error_psi_rad = traj_1.compute_error_to_traj(traj_min);

            print('Traj id: {} match traj truth id: {}'.format(traj_1.get_id(), traj_min.get_id()))
            print('Closest traj index: {}'.format(index));
            print('Result: error mean: xy:{} vxvy:{} psi_rad:{}'.format(np.mean(error_xy), np.mean(error_vxvy), np.mean(error_psi_rad)))
            print('')
            result_error_xy.append(np.mean(error_xy));
            result_error_vxvy.append(np.mean(error_vxvy));
            result_error_psi_rad.append(np.mean(error_psi_rad));


        # if math.isnan((np.mean(error))):
        #     print('Result: error: {}'.format((error)))


    for e in list(result_error_xy):
        if e > 10:
            result_error_xy.remove(e);

    result_error_xy = np.array(result_error_xy);


    error_xy = np.nanmean(result_error_xy);
    error_vxvy = np.nanmean(result_error_vxvy);
    error_psi_rad =  np.nanmean(result_error_psi_rad);
    # result_error = np.nan_to_num(result_error);
    print(result_error_xy);
    print('**************************')
    print('Average Error: xy:{} vxvy:{} psi_rad:{}'.format(error_xy, error_vxvy, error_psi_rad));
    print('Trajectories {}'.format(len(traj_list)))
    print('Total trajectories time: {}s'.format(float(total_legnth_ms)/float(1e3)))
    print('**************************')

    if display_hist:

        plt.figure('Hist xy')
        plt.hist(result_error_xy, bins=100);

        plt.figure('Hist vxvy')
        plt.hist(result_error_vxvy, bins=100);

        plt.figure('Hist psi (rad)')
        plt.hist(result_error_psi_rad, bins=100);

        plt.show()


    return error_xy, error_vxvy, error_psi_rad, result_traj_list, result_traj_truth_list;


def show_sat_img(cam_model_sat, img_sat, traj_list, traj_truth_list):

    for traj, traj_truth in zip(traj_list, traj_truth_list):

        time_ms_list = traj.get_time_ms_list();

        for time_ms in time_ms_list:

            img_sat_temp = copy.copy(img_sat);
            img_sat_temp, img_street = trajutil.plot_traj_on_images([traj], time_ms, None, img_sat_temp, None, cam_model_sat, color_text=(0,0,255));
            img_sat_temp, img_street = trajutil.plot_traj_on_images([traj_truth], time_ms, None, img_sat_temp, None, cam_model_sat, color_text=(255,0,0));

            cv2.imshow('img_sat', img_sat_temp)

            # Normal Mode: Press q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return;

            print('time_ms: {}'.format(time_ms))
            time.sleep(0.067);


def main():

    # Input csv files:
    traj_path_list = ['data/carla_precision/area1_2/output/smoothed_box3d/csv',\
                      'data/carla_precision/area2_1/output/smoothed_box3d/csv'];

    traj_path_truth_list = ['data/carla_precision/truth_csv',\
                            'data/carla_precision/truth_csv'];

    # If showing detection on igmages:
    # camera_cfg_path = '/media/caor/67c22890-fe0f-4156-9112-04f0cc358d5b/datasets/carla/carla_20190214/area2_1/carla_area2_sat_cfg.yml';
    # img_sat_path = '/media/caor/67c22890-fe0f-4156-9112-04f0cc358d5b/datasets/carla/carla_20190214/area2_1/carla_area2_sat.png';


    error_xy_list = [];
    error_vxvy_list = [];
    error_psi_rad_list = [];
    nb_traj = 0;

    for traj_path, traj_path_truth in zip(traj_path_list, traj_path_truth_list):
        print('Processing traj_path: {}'.format(traj_path));
        traj_list = trajutil.read_traj_list_from_csv(traj_path);
        traj_truth_list = trajutil.read_traj_list_from_csv(traj_path_truth);

        error_xy, error_vxvy, error_psi_rad, result_traj_list, result_traj_truth_list = compute_precision(traj_list, traj_truth_list, display_hist=False);

        # If showing detection on igmages:

        # Construct camera model
        # cam_model_sat = calib_utils.read_camera_calibration(camera_cfg_path);
        # img_sat_og = cv2.imread(img_sat_path);

        # Sow on images
        # show_sat_img(cam_model_sat, img_sat_og, result_traj_list, result_traj_truth_list);

        error_xy_list.append(error_xy);
        error_vxvy_list.append(error_vxvy);
        error_psi_rad_list.append(error_psi_rad);

        nb_traj = nb_traj+len(result_traj_list);


    error_xy = np.mean(error_xy_list);
    error_vxvy =  np.mean(error_vxvy_list);
    error_psi_rad = np.mean(error_psi_rad_list);

    print('**************************')
    print('TOTAL Average Error: xy:{} vxvy:{} psi_rad:{}'.format(error_xy, error_vxvy, error_psi_rad));
    print('Trajectories {}'.format(nb_traj))
    print('**************************')

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')