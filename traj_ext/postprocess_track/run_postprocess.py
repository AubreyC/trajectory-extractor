# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-04-05 09:50:35
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

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
import csv
import json

from traj_ext.utils import cfgutil
from traj_ext.utils.mathutil import *

from traj_ext.object_det.mask_rcnn import detect_utils

from traj_ext.tracker import EKF_utils
from traj_ext.tracker import cameramodel

from traj_ext.postprocess_track import trajutil
from traj_ext.postprocess_track import trajectory
from traj_ext.postprocess_track.time_ignore import TimeIgnore
from traj_ext.postprocess_track import track_process

from traj_ext.det_association import track_merge
from traj_ext.det_association import track_2D

from traj_ext.object_det.det_object import DetObject
from traj_ext.utils import det_zone
from traj_ext.box3D_fitting import box3D_object

from traj_ext.postprocess_track.agent_type_correct import AgentTypeCorrect

def get_tk_postprocess_in_list(tk_postprocess_list, id):

    for tk_postprocess in reversed(tk_postprocess_list):
        if id == tk_postprocess.get_id():
            return tk_postprocess;

    return None;

def main(args_input):

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
        '-image_dir',
        default='',
        help='Path of the image folder')
    argparser.add_argument(
        '-det_dir',
        default='',
        help='Path of the detection folder');
    argparser.add_argument(
        '-det_asso_dir',
        default='',
        help='Path of the detection association folder');
    argparser.add_argument(
        '-track_merge',
        default='',
        help='Path to the track merge file');
    argparser.add_argument(
        '-box3D_dir',
        default='',
        help='Path to the box3D folder');
    argparser.add_argument(
        '-camera_street',
        default='',
        help='Path to the camera street model yaml');
    argparser.add_argument(
        '-camera_sat',
        default='',
        help='Path to the camera sat model yaml');
    argparser.add_argument(
        '-camera_sat_img',
        default='',
        help='Path to the camera sat image');
    argparser.add_argument(
        '-det_zone_fned',
        default='',
        help='Path of the detection zone fned file');

    argparser.add_argument(
        '-output_dir',
        default='',
        help='Path of the output');


    argparser.add_argument(
        '-no_save_csv',
        action ='store_true',
        help='Do not save output as csv');
    argparser.add_argument(
        '-no_save_images',
        action ='store_true',
        help='Do not save output images');
    argparser.add_argument(
        '-show_images',
        action ='store_true',
        help='Show detections on images');

    argparser.add_argument(
        '-dynamic_model',
        type =str,
        default='BM2',
        help='Dynamical Model used for smoothing: CV, CVCYR, BM2');
    argparser.add_argument(
        '-delta_ms',
        type =int,
        default=100,
        help='Delta time between frames in ms');
    argparser.add_argument(
        '-box3D_minimum_overlap',
        type =float,
        default = 0.4,
        help='Minimum overlap between detection mask and estimated box3D');
    argparser.add_argument(
        '-projection_mode',
        type=str,
        default ='center_2d_height',
        help='Projection mode: \'box3D\' or \'center_2d_height\' or \'center_2d\'');
    argparser.add_argument(
        '-delete_incomplete_tracks',
        type =float,
        default = 1.0,
        help='Shrink parameter used to delete \'incomplete\' tracks: starts or ends inside the detection zone if < 1.0');

    argparser.add_argument(
        '-location_name',
        type=str,
        default ='',
        help='Location name');
    argparser.add_argument(
        '-date',
        type=str,
        default ='20190101',
        help='Date');
    argparser.add_argument(
        '-start_time',
        type=str,
        default ='0000',
        help='Start Time: HHMM');

    argparser.add_argument(
        '-config_json',
        default='',
        help='Path to json config')
    argparser.add_argument(
        '-frame_limit',
        type=int,
        default=0,
        help='Frame limit: 0 = no limit')

    args = argparser.parse_args(args_input);

    if os.path.isfile(args.config_json):
        with open(args.config_json, 'r') as f:
            data_json = json.load(f)
            vars(args).update(data_json)

    vars(args).pop('config_json', None);

    return run_postprocess_tracker(args);

def run_postprocess_tracker(config):

    # Create output folder
    output_dir = config.output_dir;
    output_dir = os.path.join(output_dir, 'traj');
    os.makedirs(output_dir, exist_ok=True)

    # Save the cfg file with the output:
    try:
        cfg_save_path = os.path.join(output_dir, 'tracker_postprocess_cfg.json');
        with open(cfg_save_path, 'w') as json_file:
            config_dict = vars(config);
            json.dump(config_dict, json_file, indent=4)
    except Exception as e:
        print('[ERROR]: Error saving config file in output folder:\n')
        print('{}'.format(e))
        return False;

    # Create output dorectory to save filtered image:
    tracker_img_dir = os.path.join(output_dir, 'img');
    os.makedirs(tracker_img_dir, exist_ok=True)
    tracker_csv_dir = os.path.join(output_dir, 'csv');
    os.makedirs(tracker_csv_dir, exist_ok=True)

    # Options for the outputs
    save_csv = not config.no_save_csv;
    save_images = not config.no_save_images;
    show_images = config.show_images
    frame_limit = config.frame_limit;

    # Open img folder
    list_img_file = os.listdir(config.image_dir);
    list_img_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))));
    if frame_limit > 0:
        list_img_file = list_img_file[:frame_limit];

    # Get name prefix
    name_prefix = '';
    if len(list_img_file) > 0:
        name_prefix = trajutil.get_name_prefix(list_img_file[0]);

    # Agent type correction
    agenttype_corr_path = os.path.join(tracker_csv_dir, name_prefix + '_traj_type_corr.csv');
    print('FILE: agent type correction: {}'.format(os.path.isfile(agenttype_corr_path)))
    # Create if does not exist
    if not os.path.isfile(agenttype_corr_path):
        AgentTypeCorrect.to_csv(agenttype_corr_path, []);
    list_agenttype_correct = AgentTypeCorrect.from_csv(agenttype_corr_path);

    # Trajectory reverse: Used in BM2 model for init in right direction
    traj_reverse_path = os.path.join(tracker_csv_dir, name_prefix + '_traj_reverse.csv');
    print('FILE: traj reverse: {}'.format(os.path.isfile(traj_reverse_path)))
    # Create if does not exist
    if not os.path.isfile(traj_reverse_path):
        trajutil.write_list_csv(traj_reverse_path, []);
    list_traj_reverse = trajutil.read_list_csv(traj_reverse_path);

    # Trajectory ignore:
    traj_ingore_path = os.path.join(tracker_csv_dir, name_prefix + '_traj_ignore.csv');
    print('FILE: traj ignore: {}'.format(os.path.isfile(traj_reverse_path)))
    # Create if does not exist
    if not os.path.isfile(traj_ingore_path):
        trajutil.write_traj_ignore_list_csv(traj_ingore_path, []);

    # Trajectory time ignore:
    time_ignore_path = os.path.join(tracker_csv_dir, name_prefix + '_time_ignore.csv');
    print('FILE: traj time ignore: {}'.format(os.path.isfile(time_ignore_path)))
    # Create if does not exist
    if not os.path.isfile(time_ignore_path):
        TimeIgnore.to_csv(time_ignore_path, []);

    # Read traj merge
    tk_merge_list = track_merge.TrackMerge.read_track_merge_csv(config.track_merge);

    ##########################################################
    # Camera Parameters
    ##########################################################

    # Camera model street
    cam_model_1 = cameramodel.CameraModel.read_from_yml(config.camera_street);

    # Camera sat
    cam_model_2 = cameramodel.CameraModel.read_from_yml(config.camera_sat);

    # ######################################################
    # Initialize parameters :
    # ######################################################
    box3D_minimum_overlap = config.box3D_minimum_overlap

    delta_ms = config.delta_ms;
    if not delta_ms > 0:
        print('\n[Error]: delta_ms is not valid: {}'.format(delta_ms));
        return False;

    # Get satellite image
    im_sat_2 = cv2.imread(config.camera_sat_img);
    if im_sat_2 is None:
        print('\n[Error]: camera_sat_img is not valid: {}'.format(config.camera_sat_img));
        return False;

    # Get mode of 2D to 3D conversion:
    projection_mode = config.projection_mode;

    ##########################################################
    # Detection zone
    ##########################################################

    # Det zone:
    det_zone_FNED = None;
    if os.path.isfile(config.det_zone_fned):
        det_zone_FNED = det_zone.DetZoneFNED.read_from_yml(config.det_zone_fned);

    # Det zone clean
    shrink_factor = config.delete_incomplete_tracks;
    det_zone_FNED_clean = None;
    if shrink_factor < 0.99:
        det_zone_FNED_clean = det_zone_FNED.shrink_zone(shrink_factor = shrink_factor)

    # ######################################################
    # Get track 2D from detection, detection association
    # and if available box3D
    # ######################################################

    det_folder_path = config.det_dir;
    det_asso_folder_path = config.det_asso_dir;
    box3D_folder_path = config.box3D_dir;

    track_2D_list, frame_index_list = track_2D.Track2D.from_csv(list_img_file, det_folder_path, det_asso_folder_path, box3D_folder_path);

    # Create TrackProcess from track2D
    track_process_list = [];
    for tk_2D in track_2D_list:
        tk_process = track_process.TrackProcess(tk_2D, cam_model_1, dynamic_model = config.dynamic_model, projection_mode = projection_mode, box3D_minimum_overlap = box3D_minimum_overlap);
        track_process_list.append(tk_process);

    # ######################################################
    # Post-Process each tracks
    # ######################################################

    # Get list of time_ms
    list_times_ms = [int(x * delta_ms) for x in frame_index_list];

    # Filter and smooth all trajectories
    for tk_process in track_process_list:

        reverse_init_BM = False;
        if tk_process.get_id() in list_traj_reverse:
            reverse_init_BM = True;

        tk_process.process_traj(list_times_ms, reverse_init_BM = reverse_init_BM);

    # ######################################################
    # Clean trajectories
    # ######################################################

    # remove tracks
    for tk_process in list(track_process_list):
        if (tk_process.get_trajectory() is None):
            track_process_list.remove(tk_process);
            print('Removing traj: {} None'.format(tk_process.get_id()));

    for tk_process in list(track_process_list):

        # With few measurments
        if tk_process.track_2D.get_length() < 1:
            print('Removing traj: {} traj nb meas: {} ms'.format(tk_process.get_id(), tk_process.track_2D.get_length()));
            track_process_list.remove(tk_process);

    print('[INFO]: Number of track not merged: {}'.format(len(track_process_list)));

    # ######################################################
    # Write trajectories
    # ######################################################

    # In csv
    trajectory_list = [];
    for tk_process in track_process_list:

        if not(tk_process.get_trajectory() is None):
            trajectory_list.append(tk_process.get_trajectory());

    # With panda
    name_prefix_not_merged = name_prefix + '_traj_not_merged';
    trajectory.Trajectory.write_trajectory_panda_csv(tracker_csv_dir, name_prefix_not_merged, trajectory_list, list_times_ms);

    # ######################################################
    # Merge trajectories
    # ######################################################

    new_tk_list = [];
    for new_tk_post in tk_merge_list:

        new_tk = None;
        for tk_id in new_tk_post:
            if new_tk is None:
                new_tk = get_tk_postprocess_in_list(track_process_list, tk_id);

                if new_tk:
                    track_process_list.remove(new_tk);


            else:
                next_tk = get_tk_postprocess_in_list(track_process_list, tk_id);
                if not (next_tk is None):
                    # print('new: {} next: {}'.format(new_tk, next_tk))
                    new_tk.track_2D.merge_with_track2D(next_tk.track_2D);

                    track_process_list.remove(next_tk);


        new_tk_list.append(new_tk);

    for tk_new in new_tk_list:

        if not (tk_new is None):
            # tk_new.process_traj(list_times_ms);

            track_process_list.append(tk_new);

    # ######################################################
    # Replace wrong agent type
    # ######################################################

    # Read agent corrections
    for agenttype_correct in list_agenttype_correct:

        print('Correcting track_id: {} agent_type: {}'.format(agenttype_correct.track_id, agenttype_correct.agent_type));
        tk = get_tk_postprocess_in_list(track_process_list, agenttype_correct.track_id);
        if not (tk is None):
            print('Correct track_id: {} agent_type: {}'.format(tk.get_id(), agenttype_correct.agent_type));
            tk.set_agent_type(agenttype_correct.agent_type);

    # ######################################################
    # Post-Process each tracks
    # ######################################################

    # Filter and smooth all trajectories
    for tk_process in track_process_list:

        reverse_init_BM = False;
        if tk_process.get_id() in list_traj_reverse:
            reverse_init_BM = True;

        tk_process.process_traj(list_times_ms, reverse_init_BM = reverse_init_BM);

    # ==========================================
    #         Clean trajectories
    # ==========================================

    for tk_postprocess in list(track_process_list):
        if not (tk_postprocess.get_trajectory() is None):

            tk_postprocess.get_trajectory().delete_point_outside(det_zone_FNED);

    # remove tracks
    for tk_postprocess in list(track_process_list):
        if (tk_postprocess.get_trajectory() is None):
            track_process_list.remove(tk_postprocess);
            print('Removing traj: {} None'.format(tk_postprocess.get_id()));


    print('[INFO]: Number of track merged: {}'.format(len(track_process_list)));

    for tk_postprocess in list(track_process_list):

        # With few measurments
        if tk_postprocess.get_trajectory().get_length() < 1:

            print('Removing traj: {} traj length: {}'.format(tk_postprocess.get_id(), tk_postprocess.get_trajectory().get_length()));
            track_process_list.remove(tk_postprocess);

    print('[INFO]: Number of track merged: {}'.format(len(track_process_list)));

    # ==========================================
    #         Clean trajectories
    # ==========================================

    # Remove tracks that starts of end in the middle of the detection zone:
    if not (det_zone_FNED_clean is None):

        for tk_postprocess in list(track_process_list):

            traj = tk_postprocess.get_trajectory();

            if not (traj is None):
                trajoint_start = traj.get_start_trajoint();
                trajoint_end = traj.get_end_trajoint();

                start_in = det_zone_FNED_clean.in_zone(np.array([[trajoint_start.x], [trajoint_start.y]]));
                end_in = det_zone_FNED_clean.in_zone(np.array([[trajoint_end.x], [trajoint_end.y]]));

                if start_in or end_in:
                    track_process_list.remove(tk_postprocess);
                    print('Removing traj: {} start_in: {} end_in: {}'.format(tk_postprocess.get_id(), start_in, end_in));

    print('[INFO]: Number of track: {}'.format(len(track_process_list)));

    # ==========================================
    #         Write trajectory csv
    # ==========================================

    # In csv
    trajectory_list = [];
    for tk_postprocess in track_process_list:

        if not(tk_postprocess.get_trajectory() is None):
            trajectory_list.append(tk_postprocess.get_trajectory());

    # With panda
    trajectory.Trajectory.write_trajectory_panda_csv(tracker_csv_dir, name_prefix + '_traj', trajectory_list, list_times_ms);
    trajutil.write_time_list_csv(tracker_csv_dir, name_prefix, list_times_ms);


    total_traj_nb, total_traj_time_s, total_traj_distance_m, duration_s = trajectory.Trajectory.generate_metadata(trajectory_list, list_times_ms);

    trajutil.write_trajectory_meta(tracker_csv_dir, name_prefix + '_traj', config.location_name,\
                                                        config.date,\
                                                        config.start_time,\
                                                        duration_s,
                                                        config.delta_ms,
                                                        total_traj_nb,\
                                                        total_traj_time_s,\
                                                        total_traj_distance_m);

    # ==========================================
    #         Plot Filter and Smoother
    # ==========================================
    if show_images or save_images:

        for current_index, current_img_file in enumerate(list_img_file):

            # Means that we have reached the end of the processing timeframe
            if current_index >= len(list_times_ms):
                break;

            t_smooth_ms = list_times_ms[current_index];

            # Getting current street image
            img_street = cv2.imread(os.path.join(config.image_dir, current_img_file));
            print('current_img_file: {}'.format(current_img_file))

            # Copy to get a clean image to show detection for each frame
            img_sat = copy.copy(im_sat_2);

            # Draw tracking zone:
            if not (det_zone_FNED is None):

                det_zone_FNED.display_on_image(img_sat, cam_model_2, thickness=1);
                det_zone_FNED.display_on_image(img_street, cam_model_1, thickness=1);

                if not (det_zone_FNED_clean is None):
                    det_zone_FNED_clean.display_on_image(img_sat, cam_model_2, color=(0,100,255), thickness=1);
                    det_zone_FNED_clean.display_on_image(img_street, cam_model_1, color=(0,100,255), thickness=1);

            img_street = trajutil.display_traj_list_on_image(trajectory_list, t_smooth_ms, img_street, cam_model_1);
            img_sat = trajutil.display_traj_list_on_image(trajectory_list, t_smooth_ms, img_sat, cam_model_2);

            # Use to display measurements
            # for tk_postprocess in track_process_list:
            #     tk_postprocess.track_2D.display_on_image(current_index, img_street, cam_model_1);

            # Concatenate images for easier display
            image_tracker = EKF_utils.concatenate_images(img_street, img_sat);

            print('[Info]: Plotting step: {}'.format(current_index))

            # Save images if needed
            if save_images:
                img_track_name = current_img_file.split('.')[0] + "_filter.png"
                img_track_path = os.path.join(tracker_img_dir, img_track_name);

                cv2.imwrite(img_track_path, image_tracker);


            if show_images:
                cv2.imshow('PostProcess tracker', image_tracker)

                # Normal Mode: Press q to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break;

    return True;

if __name__ == '__main__':

    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')