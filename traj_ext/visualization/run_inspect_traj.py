# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-24 15:34:17
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-28 21:53:43

import copy
import cv2
import argparse
import os
import subprocess
import math;
import sys
import json

from traj_ext.postprocess_track import trajutil
from traj_ext.postprocess_track.trajectory import Trajectory

from traj_ext.tracker.cameramodel import CameraModel

from traj_ext.postprocess_track.agent_type_correct import AgentTypeCorrect

from traj_ext.object_det.det_object import DetObject
from traj_ext.object_det.mask_rcnn import detect_utils

from traj_ext.hd_map.HD_map import HDmap
from traj_ext.object_det import run_create_det_object

from traj_ext.postprocess_track.time_ignore import TimeIgnore


from traj_ext.utils import det_zone
from traj_ext.utils import mathutil


def export_trajectories(list_times_ms, traj_list, list_traj_ignore_list=[[]],time_ignore_list=[], det_zone_ignore_list=[], name_sequence='', traj_saving_dir='', location_name='', date='', start_time='', delta_ms=100):

    trajectory_list_clean = copy.copy(traj_list);
    for traj in list(trajectory_list_clean):

        # Only plot trajectory not ignore
        for list_traj_ignore in list_traj_ignore_list:

            if (traj.get_id() in list_traj_ignore):
                trajectory_list_clean.remove(traj);
                print('Removing : {}'.format(traj.get_id()))

    # Remove points that are in the det_ignore list
    for traj in trajectory_list_clean:
        traj.remove_point_in_time_ignore(time_ignore_list, det_zone_ignore_list);

    print('Exporting trajectory: folder {} file: {}'.format(traj_saving_dir, name_sequence + '.csv'))

    Trajectory.write_trajectory_panda_csv(traj_saving_dir, name_sequence + '_traj', trajectory_list_clean, list_times_ms);
    trajutil.write_time_list_csv(traj_saving_dir, name_sequence, list_times_ms);


    # Write meta data
    total_traj_nb, total_traj_time_s, total_traj_distance_m, duration_s = Trajectory.generate_metadata(trajectory_list_clean, list_times_ms);
    print('Total traj: {} Total time: {}s Total distance: {}m'.format(total_traj_nb, total_traj_time_s, total_traj_distance_m));

    trajutil.write_trajectory_meta(traj_saving_dir, name_sequence + '_traj' , location_name,\
                                                          date,\
                                                          start_time,\
                                                          duration_s,
                                                          str(delta_ms),
                                                          total_traj_nb,\
                                                          total_traj_time_s,\
                                                          total_traj_distance_m);

def display_traj_on_image(time_ms, cam_model, image, traj_list, traj_ignore_list_1 = [], traj_ignore_list_2 = [], traj_ignore_list_3 = [], det_zone_FNED_list = [], no_label = False, display_traj = True, only_complete = False, complete_marker = False, velocity_label = False):
    """Display the trajectory on an image

    Args:
        time_ms (TYPE): Description
        cam_model (TYPE): Description
        image (TYPE): Description
        traj_list (TYPE): Description
        traj_ignore_list_1 (list, optional): Description
        traj_ignore_list_2 (list, optional): Description
        det_zone_FNED_list (list, optional): Description
        only_complete (bool, optional): Description

    Returns:
        TYPE: Description
    """
    # Copy image to avoid modifying og image
    image_current = copy.copy(image);

    for det_zone_FNED in det_zone_FNED_list:

        if not det_zone_FNED is None:
            # Draw detection zone
            image_current = det_zone_FNED.display_on_image(image_current, cam_model);

    # Plot trajectory
    det_track_list = [];
    if display_traj:
        for traj in traj_list:

            # Only plot trajectory not ignore
            if not (traj.get_id() in traj_ignore_list_1) and not (traj.get_id() in traj_ignore_list_2) and not (traj.get_id() in traj_ignore_list_3):

                image_current, det_track = traj.display_on_image(time_ms, image_current, cam_model, only_complete = only_complete, complete_marker = complete_marker, no_label = no_label, velocity_label = velocity_label);
                if not det_track is None:
                    det_track_list.append(det_track);

    return image_current, det_track_list;


def click_detection(event, x, y, flags, param):
    """Click callback to enable / disbale specific detections by clicking on it

    Args:
        event (TYPE): Description
        x (TYPE): Description
        y (TYPE): Description
        flags (TYPE): Description
        param (TYPE): Description

    Returns:
        TYPE: None
    """

    # If clicked
    if event == cv2.EVENT_LBUTTONDOWN:

        # Get parameters
        det_object_list = param[0];
        det_csv_path = param[1];
        image_current = param[2];
        track_id_text = param[3];
        label_replace = param[4];

        # Copy current image
        image_current_det = copy.copy(image_current);

        # Get current detections
        # det_object_list = DetObject.from_csv(det_csv_path);

        # Enable / Disable detection that corresponds to the click
        det_modif = False;
        for det_object in det_object_list:
            if det_object.is_point_in_det_2Dbox(x, y):

                det_object.good = not det_object.good;
                det_modif = True;
                print('Detection {}: {}'.format(det_object.det_id, det_object.good));

        # Save the detctions to csv
        if det_modif:
            print('Saving detections: {}'.format(det_csv_path));
            DetObject.to_csv(det_csv_path, det_object_list);

        # Show new detections
        for det in det_object_list:
            det.display_on_image(image_current_det, track_id_text = track_id_text);
        cv2.imshow('Detection', image_current_det)

    # # If clicked
    if event == cv2.EVENT_RBUTTONDOWN:

        # Get parameters
        det_object_list = param[0];
        det_csv_path = param[1];
        image_current = param[2];
        track_id_text = param[3];
        label_replace = param[4];

        # Copy current image
        image_current_det = copy.copy(image_current);

        # Get current detections
        # det_object_list = DetObject.from_csv(det_csv_path);

        # Enable / Disable detection that corresponds to the click
        for det_object in det_object_list:
            if det_object.is_point_in_det_2Dbox(x, y):

                det_object.label = label_replace;
                print('Detection {}: {}'.format(det_object.det_id, det_object.label));

        # Save the detctions to csv
        print('Saving detections: {}'.format(det_csv_path));
        DetObject.to_csv(det_csv_path, det_object_list);

        # Show new detections
        for det in det_object_list:
            det.display_on_image(image_current_det, track_id_text = track_id_text);
        cv2.imshow('Detection', image_current_det)


    return;



def get_track_id_image( x,y, det_object_track_list):
    """Get the track ID based on a position on image

    Args:
        x (TYPE): x pixel position
        y (TYPE): y pixel position
        det_object_track_list (TYPE): detection object list

    Returns:
        TYPE: track id
    """
    track_id = -1;
    for det_object in det_object_track_list:
        if det_object.is_point_in_det_2Dbox(x, y):

            track_id = det_object.track_id;

            print('Track id: {}'.format(det_object.track_id));

    return track_id;


def click_hd_merged(event, x, y, flags, param):
    """Callback for HD view of track merged

    Args:
        event (TYPE): Description
        x (TYPE): Description
        y (TYPE): Description
        flags (TYPE): Description
        param (TYPE): Description

    Returns:
        TYPE: Description
    """
    # If clicked
    if event == cv2.EVENT_LBUTTONDOWN:

        # Get parameters
        det_object_track_list = param[0];

        track_id = get_track_id_image(x, y, det_object_track_list);

    return;

def click_hd_not_merged(event, x, y, flags, param):
    """Callback for HD view of track not merged

    Args:
        event (TYPE): Description
        x (TYPE): Description
        y (TYPE): Description
        flags (TYPE): Description
        param (TYPE): Description

    Returns:
        TYPE: Description
    """

    # If clicked
    if event == cv2.EVENT_LBUTTONDOWN:

        # Get parameters
        det_object_track_list = param[0];

        track_id = get_track_id_image(x, y, det_object_track_list);

    return;


def print_instructions():

    print('\nInstuctions Trajectories Inspection:\
                         \n- n: Next frame\
                         \n- b: Jump 1000 frame forward\
                         \n- p: Previous frame\
                         \n- b: Jump 1000 frame backward\
                         \n- +: Increase skip value\
                         \n- -: Decrease skip value\
                         \n- d: Open detection window\
                         \n- a: Open adding detection window\
                         \n- w: Open detection and track_id files\
                         \n- c: Open Agent type correction\
                         \n- m: Open merging file with sublime text\
                         \n- s: Enable saving form current frame\
                         \n- Click on Detection window: Enable/disable detections\
                         \n- f: Display only complete trajectories\
                         \n- t: Open time ignore file\
                         \n- i: Open ignore trajectory file\
                         \n- r: Open reverse trajectory file\
                         \n- e: Export trajectory file\
                         \n- z: Show / Hide satellite view\
                         \n- esc: Quit\
                         \n')


def main(args_input):

    # Print instructions
    print("############################################################")
    print("Inspect trajectories and export clean trajectories")
    print("############################################################\n")

    # ##########################################################
    # # Parse Arguments
    # ##########################################################
    argparser = argparse.ArgumentParser(
        description='Inspect trajectories and export clean trajectories')
    argparser.add_argument(
        '-traj',
        default='',
        help='Path of the trajectories csv')
    argparser.add_argument(
        '-traj_not_merged',
        default='',
        help='Path of the trajectories not merged csv')
    argparser.add_argument(
        '-traj_ignore',
        default='',
        help='Path of the trajectories ignore csv')
    argparser.add_argument(
        '-traj_reverse',
        default='',
        help='Path of the trajectories reverse csv')
    argparser.add_argument(
        '-traj_type_corr',
        default='',
        help='Path of the type correction csv')
    argparser.add_argument(
        '-traj_time_ignore',
        default='',
        help='Path of the traj time ignore csv')
    argparser.add_argument(
        '-time',
        default='',
        help='Path of the time csv')

    argparser.add_argument(
        '-image_dir',
        type=str,
        default='',
        help='Path of the image folder')
    argparser.add_argument(
        '-det_dir',
        type=str,
        default='',
        help='Path of the detection folder');
    argparser.add_argument(
        '-det_asso_dir',
        type=str,
        default='',
        help='Path of the detection association folder');
    argparser.add_argument(
        '-track_merge',
        type=str,
        default='',
        help='Path to the track merge file');
    argparser.add_argument(
        '-camera_street',
        type=str,
        default='',
        help='Path to the camera street model yaml');
    argparser.add_argument(
        '-camera_sat',
        type=str,
        default='',
        help='Path to the camera sat model yaml');
    argparser.add_argument(
        '-camera_sat_img',
        type=str,
        default='',
        help='Path to the camera sat image');
    argparser.add_argument(
        '-det_zone_fned',
        type=str,
        default='',
        help='Path of the detection zone fned file');
    argparser.add_argument(
        '-shrink_zone',
        type = float,
        default = 1.0,
        help='Detection zone shrink coefficient for complete trajectories');
    argparser.add_argument(
        '-min_length',
        type = int,
        default = 5,
        help='Ignore trajectory that have fewer points than this value');
    argparser.add_argument(
        '-label_replace',
        type = str,
        default = 'car',
        help='Label used to replace labels');
    argparser.add_argument(
        '-hd_map',
        type = str,
        default = '',
        help='Path to the HD map');
    argparser.add_argument(
        '-det_zone_ignore',
        type = str,
        nargs = '*',
        default = [],
        help='Path to detection zone ignore');
    argparser.add_argument(
        '-export',
        action ='store_true',
        help='Export trajectories directly and exit the program (for automated process)');

    argparser.add_argument(
        '-output_dir',
        type=str,
        default='',
        help='Path of the output');


    argparser.add_argument(
        '-location_name',
        type=str,
        default ='',
        help='Location name');
    argparser.add_argument(
        '-date',
        type=str,
        default ='',
        help='Date');
    argparser.add_argument(
        '-start_time',
        type=str,
        default ='0000',
        help='Start Time: HHMM');
    argparser.add_argument(
        '-delta_ms',
        type =int,
        default=100,
        help='Delta time between frames in ms');

    argparser.add_argument(
        '-config_json',
        default='',
        help='Path to json config')

    args = argparser.parse_args(args_input);

    if os.path.isfile(args.config_json):
        with open(args.config_json, 'r') as f:
            data_json = json.load(f)
            vars(args).update(data_json)

    vars(args).pop('config_json', None);

    run_inspect_traj(args);

def run_inspect_traj(config):

    # Create output folder
    output_dir = config.output_dir;
    output_dir = os.path.join(output_dir, 'traj_inspect');
    os.makedirs(output_dir, exist_ok=True)

    # Create output sub-folder
    image_raw_saving_dir = os.path.join(output_dir, 'img_raw');
    image_annoted_saving_dir = os.path.join(output_dir, 'img_annoted');
    image_hd_map_saving_dir = os.path.join(output_dir, 'img_hdmap');
    traj_saving_dir = os.path.join(output_dir, 'csv');

    os.makedirs(image_raw_saving_dir, exist_ok=True)
    os.makedirs(image_annoted_saving_dir, exist_ok=True)
    os.makedirs(image_hd_map_saving_dir, exist_ok=True)
    os.makedirs(traj_saving_dir, exist_ok=True)

    # Save the cfg file with the output:
    try:
        cfg_save_path = os.path.join(output_dir, 'inspect_traj_cfg.json');
        with open(cfg_save_path, 'w') as json_file:
            config_dict = vars(config);
            json.dump(config_dict, json_file, indent=4)
    except Exception as e:
        print('[ERROR]: Error saving config file in output folder:\n')
        print('{}'.format(e))
        return;


    # Check if det data available:
    det_data_available = os.path.isdir(config.det_dir);
    print('Detection data: {}'.format(det_data_available));

    det_asso_data_available = os.path.isdir(config.det_asso_dir);
    print('Detection Association data: {}'.format(det_asso_data_available));

    track_merge_data_available = os.path.isfile(config.track_merge);
    print('Detection Merge data: {}'.format(track_merge_data_available));

    # Check if trajectory ingore exist:
    traj_ingore_path = config.traj_ignore;
    if traj_ingore_path == '':
        traj_ingore_path = config.traj.replace('traj.csv', 'traj_ignore.csv');

    if not os.path.isfile(traj_ingore_path):
        print('[ERROR]: Trajectory ignore file not found: {}'.format(traj_ingore_path))
        return;
    list_traj_ignore = trajutil.read_traj_ignore_list_csv(traj_ingore_path);

    # Trajectory reverse exist:
    traj_reverse_path = config.traj_reverse
    if traj_reverse_path == '':
        traj_reverse_path = config.traj.replace('traj.csv', 'traj_reverse.csv');

    if not os.path.isfile(traj_reverse_path):
        print('[ERROR]: Trajectory reverse file not found: {}'.format(traj_reverse_path))
        return;

    # Agent type correction
    agenttype_corr_path = config.traj_type_corr;
    if agenttype_corr_path == '':
        agenttype_corr_path = config.traj.replace('traj.csv', 'traj_type_corr.csv');

    if not os.path.isfile(agenttype_corr_path):
        print('[ERROR]: Agent type correction file not found: {}'.format(agenttype_corr_path))
        return;
    list_agenttype_correct = AgentTypeCorrect.from_csv(agenttype_corr_path);

    # Time ignore
    time_ignore_path = config.traj_time_ignore;
    if time_ignore_path == '':
        time_ignore_path = config.traj.replace('traj.csv', 'time_ignore.csv');

    if not os.path.isfile(time_ignore_path):
        print('[ERROR]: Traj time ignore file not found: {}'.format(time_ignore_path))
        return;
    time_ignore_list = TimeIgnore.from_csv(time_ignore_path);

    # Time
    list_times_ms_path = config.time;
    if list_times_ms_path == '':
        list_times_ms_path = config.traj.replace('traj.csv', 'time_traj.csv');
    if not os.path.isfile(list_times_ms_path):
        print('[ERROR]: Traj time file not found: {}'.format(list_times_ms_path))
        return;
    list_times_ms = trajutil.read_time_list_csv(list_times_ms_path);

    # Trajectory file
    if not os.path.isfile(config.traj):
        print('[ERROR]: Traj file not found: {}'.format(config.traj))
        return;
    print('Reading trajectories:');
    traj_list = Trajectory.read_trajectory_panda_csv(config.traj);

    # Traj not merged
    traj_not_merged_csv_path = config.traj_not_merged;
    if traj_not_merged_csv_path == '':
        traj_not_merged_csv_path = config.traj.replace('traj.csv', 'traj_not_merged.csv');

    if not os.path.isfile(traj_not_merged_csv_path):
        print('[ERROR]: Traj not merged file not found: {}'.format(traj_not_merged_csv_path))
        return;
    print('Reading trajectories not merged:');
    traj_list_not_merged = Trajectory.read_trajectory_panda_csv(traj_not_merged_csv_path);

    # Open objects
    cam_model = CameraModel.read_from_yml(config.camera_street);
    det_zone_FNED = det_zone.DetZoneFNED.read_from_yml(config.det_zone_fned);

    # Ignore trajectories that are smaller than min_length
    list_traj_ignore_automatic = [];
    for traj in list(traj_list):
        if traj.get_length() < config.min_length:
            if not (traj.get_id() in list_traj_ignore):
                list_traj_ignore_automatic.append(traj.get_id());
                print('Ignoring Traj {} length {}'.format(traj.get_id(), traj.get_length()))

    sat_view_available = False;
    sat_view_enable = False;
    if os.path.isfile(config.camera_sat_img) and os.path.isfile(config.camera_sat):
        cam_model_sat = CameraModel.read_from_yml(config.camera_sat);

        image_sat = cv2.imread(os.path.join(config.camera_sat_img));
        sat_view_available = True;

    # Check if image is directoty
    image_in_dir = os.path.isdir(config.image_dir);

    # If image directory, open list of images
    if image_in_dir:
        list_img_file = os.listdir(config.image_dir);
        list_img_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))));

    # Else open the unique image
    else:
        image = cv2.imread(config.image_dir);

    hd_map_mode = False;
    if os.path.isfile(config.hd_map):
        hd_map_mode = True;
        hd_map = HDmap.from_csv(config.hd_map);

        cam_model_hdmap, image_hdmap = hd_map.create_view();

        image_hdmap = hd_map.display_on_image(image_hdmap, cam_model_hdmap);

    # Shrink det zone for complete
    det_zone_FNED_complete = None;
    if config.shrink_zone < 1:
        det_zone_FNED_complete = det_zone_FNED.shrink_zone(config.shrink_zone);
        for traj in traj_list:
            if not traj.check_is_complete(det_zone_FNED_complete):

                if not (traj.get_id() in list_traj_ignore) and not (traj.get_id() in list_traj_ignore_automatic):

                    print('Track: {} time_ms: {} not complete'.format(traj.get_id(), traj.get_start_trajoint().time_ms))
    elif config.shrink_zone > 1:
        det_zone_FNED_complete = det_zone_FNED.shrink_zone(config.shrink_zone);

    # Get name sequence
    name_sequence = os.path.basename(config.traj);
    name_sequence = name_sequence.split('.')[0];
    name_sequence = name_sequence.replace('_traj','');

    # Remove in the det_zone ignore
    det_zone_ignore_list = [];

    for det_zone_ignore_path in config.det_zone_ignore:
        if  os.path.isfile(det_zone_ignore_path):
            det_zone_FNED_ignore = det_zone.DetZoneFNED.read_from_yml(det_zone_ignore_path);
            det_zone_ignore_list.append(det_zone_FNED_ignore);

    only_complete = False;
    skip_value = 1;
    det_view_enable = False;
    frame_index = 0;
    saving_mode = False;

    print_instructions();
    while True:

        #######################################################
        ## Update Information
        #######################################################

        # Read traj ignore list:
        time_ignore_list = TimeIgnore.from_csv(time_ignore_path);
        list_traj_ignore_timeignore = [];
        # if len(time_ignore_list) > 0:
        #     for traj in traj_list:
        #         if traj.check_startend_time_ignore(time_ignore_list):
        #             list_traj_ignore_timeignore.append(traj.get_id());



        #######################################################
        ## Update Information
        #######################################################

        # Read traj ignore list:
        list_traj_ignore = trajutil.read_traj_ignore_list_csv(traj_ingore_path);

        # Read agent corrections
        list_agenttype_correct = AgentTypeCorrect.from_csv(agenttype_corr_path);
        for agenttype_correct in list_agenttype_correct:
            traj = trajutil.get_traj_in_list(traj_list, agenttype_correct.track_id);
            if not (traj is None):
                traj.set_agent_type(agenttype_correct.agent_type);


        #######################################################
        ## Show Trajectories
        #######################################################

        # Get Time ms
        frame_index = mathutil.clip(frame_index, 0, len(list_img_file)-1);
        frame_index = mathutil.clip(frame_index, 0, len(list_times_ms)-1);
        time_ms = list_times_ms[frame_index];

        if image_in_dir:
            img_file_name = list_img_file[frame_index];
            image_current = cv2.imread(os.path.join(config.image_dir, img_file_name));
        else:
            # Copy image
            image_current = image;


        if not (image_current is None):

            print('Showing: frame_id: {} time_ms: {} image: {}'.format(frame_index, time_ms, img_file_name));

            display_traj = not TimeIgnore.check_time_inside_list(time_ignore_list, time_ms);

            # Display traj
            det_zone_FNED_list = [det_zone_FNED, det_zone_FNED_complete] + det_zone_ignore_list;
            image_current_traj, det_track_list_current = display_traj_on_image(time_ms, cam_model, image_current, traj_list, display_traj = display_traj, traj_ignore_list_1 = list_traj_ignore, traj_ignore_list_2 = list_traj_ignore_automatic, traj_ignore_list_3 = list_traj_ignore_timeignore, det_zone_FNED_list = det_zone_FNED_list, complete_marker = config.shrink_zone < 1);

            # Show image
            cv2.imshow('Trajectory visualizer', image_current_traj)
            cv2.setMouseCallback("Trajectory visualizer", click_hd_merged, param=[det_track_list_current])


            # Display traj
            image_current_traj_not_merged, det_track_list_not_merged = display_traj_on_image(time_ms, cam_model, image_current, traj_list_not_merged, det_zone_FNED_list = det_zone_FNED_list);

            # Show image
            cv2.imshow('Trajectory not merged', image_current_traj_not_merged)
            cv2.setMouseCallback("Trajectory not merged", click_hd_merged, param=[det_track_list_not_merged])

            if sat_view_available:

                if sat_view_enable:

                    # Display traj
                    image_sat_current_not_merged,_ = display_traj_on_image(time_ms, cam_model_sat, image_sat, traj_list_not_merged, det_zone_FNED_list = det_zone_FNED_list);

                    # Show image
                    cv2.imshow('Sat View not merged', image_sat_current_not_merged)


                    # Display traj
                    image_sat_current,_ = display_traj_on_image(time_ms, cam_model_sat, image_sat, traj_list, display_traj = display_traj, traj_ignore_list_1 = list_traj_ignore, traj_ignore_list_2 = list_traj_ignore_automatic, traj_ignore_list_3 = list_traj_ignore_timeignore, det_zone_FNED_list = det_zone_FNED_list);

                    # Show image
                    cv2.imshow('Sat View merged', image_sat_current)

            if hd_map_mode:

                # Display traj
                image_hdmap_current, det_track_list_hd_merged = display_traj_on_image(time_ms, cam_model_hdmap, image_hdmap, traj_list, display_traj = display_traj, traj_ignore_list_1 = list_traj_ignore, traj_ignore_list_2 = list_traj_ignore_automatic, traj_ignore_list_3 = list_traj_ignore_timeignore, det_zone_FNED_list = det_zone_FNED_list, complete_marker = config.shrink_zone < 1);

                # Show image
                cv2.imshow('HD map view merged', image_hdmap_current)

                cv2.setMouseCallback("HD map view merged", click_hd_merged, param=[det_track_list_hd_merged])


                # Display traj
                image_hdmap_current_not_merged, det_track_list_hd_not_merged = display_traj_on_image(time_ms, cam_model_hdmap, image_hdmap, traj_list_not_merged, det_zone_FNED_list = det_zone_FNED_list);

                # Show image
                cv2.imshow('HD map view not merged', image_hdmap_current_not_merged)

                cv2.setMouseCallback("HD map view not merged", click_hd_not_merged, param=[det_track_list_hd_not_merged])

            if saving_mode:

                img_annoted_name = img_file_name.split('.')[0] + '_annotated.png';
                print('Saving: frame_id: {} image: {}'.format(frame_index, img_annoted_name))

                image_annoted_path = os.path.join(image_annoted_saving_dir, img_annoted_name);
                cv2.imwrite(image_annoted_path, image_current_traj);

                print('Saving: frame_id: {} image: {}'.format(frame_index, img_file_name))
                image_raw_path = os.path.join(image_raw_saving_dir, img_file_name);
                cv2.imwrite(image_raw_path, image_current);

                if hd_map_mode:

                    img_hdmap_name = img_file_name.split('.')[0] + '_hdmap.png';
                    print('Saving: frame_id: {} image: {}'.format(frame_index, img_hdmap_name))
                    image_hdmap_path = os.path.join(image_hd_map_saving_dir, img_hdmap_name);
                    cv2.imwrite(image_hdmap_path, image_hdmap_current);

            #######################################################
            ## Show Detections
            #######################################################

            det_csv_path = '';
            if det_view_enable and det_data_available:
                image_current_det = copy.copy(image_current);

               # CSV name management
                det_csv_name = img_file_name.split('.')[0] + '_det.csv';
                # det_csv_path = os.path.join(config.det_dir, 'csv');
                det_csv_path = config.det_dir;
                det_csv_path = os.path.join(det_csv_path, det_csv_name);

                det_object_list = DetObject.from_csv(det_csv_path, expand_mask = True);

                track_id_text = False;
                if det_asso_data_available:
                    track_id_text = True;
                    det_asso_csv_name = img_file_name.split('.')[0] + '_detassociation.csv';
                    # det_asso_csv_path = os.path.join(config.det_asso_dir, 'csv')
                    det_asso_csv_path = config.det_asso_dir;
                    det_asso_csv_path = os.path.join(det_asso_csv_path, det_asso_csv_name);

                    try:
                        det_asso_list = detect_utils.read_dict_csv(det_asso_csv_path);
                    except Exception as e:
                        print("ERROR: Could not open csv: {}".format(e));
                        det_asso_list = [];

                    for det_object in det_object_list:

                        for det_asso in det_asso_list:

                            track_id = det_asso['track_id'];
                            det_id = det_asso['det_id'];

                            if det_id == det_object.det_id:
                                det_object.track_id = track_id;

                for det in det_object_list:
                    det.display_on_image(image_current_det, track_id_text = track_id_text);

                # Show image detection
                cv2.imshow('Detection', image_current_det)

                # Set callback to enable / disable detections
                cv2.setMouseCallback("Detection", click_detection, param=[det_object_list, det_csv_path, image_current, track_id_text, config.label_replace])

        else:
            print('Not found: frame_id: {} image: {}'.format(frame_index, img_file_name));


        if config.export:
            # Export trajectorie
            export_trajectories(list_times_ms,
                                traj_list,
                                list_traj_ignore_list=[list_traj_ignore, list_traj_ignore_automatic, list_traj_ignore_timeignore],
                                time_ignore_list=time_ignore_list,
                                det_zone_ignore_list=det_zone_ignore_list,
                                name_sequence=name_sequence,
                                traj_saving_dir=traj_saving_dir,
                                location_name=config.location_name,
                                date=config.date,
                                start_time=config.start_time,
                                delta_ms=config.delta_ms);

            #Exit the program
            break;

        #######################################################
        ## Control keys
        #######################################################
        key = cv2.waitKey(0) & 0xFF


        if key == ord("n"):
            frame_index +=skip_value;
            mathutil.clip(frame_index, 0, len(list_times_ms));

        elif key == ord("b"):
            frame_index +=1000*skip_value;
            mathutil.clip(frame_index, 0, len(list_times_ms));

        elif  key == ord("p"):
            frame_index -=skip_value;
            mathutil.clip(frame_index, 0, len(list_times_ms));

        elif  key == ord("o"):
            frame_index -=1000*skip_value;
            mathutil.clip(frame_index, 0, len(list_times_ms));

        # Only complete traj
        elif key == ord("f"):
            if config.shrink_zone < 1:
                only_complete = not only_complete;
                print('Mode: Display complete trajectroy only: {}'.format(only_complete));

        # Escape: Quit the program
        elif key == 27:
            break;

        # Escape: Quit the program
        elif key == ord('z'):
            if sat_view_available:
                if sat_view_enable:
                    cv2.destroyWindow('Sat View not merged');
                    cv2.destroyWindow('Sat View merged');

                    sat_view_enable = False;

                else:
                    sat_view_enable = True;

        elif key == ord("+"):
            skip_value +=1;
            skip_value = max(1, skip_value);
            print('Skip value: {}'.format(skip_value))

        elif key == ord("-"):
            skip_value -=1;
            skip_value = max(1, skip_value);
            print('Skip value: {}'.format(skip_value))

        elif key == ord("d"):
            if det_data_available:

                if det_view_enable:
                    cv2.destroyWindow('Detection');
                    det_view_enable = False;

                else:
                    det_view_enable = True;

            else:
                print('[Error]: Detection data not available in: {}'.format(config.det_dir));

        elif key == ord("a"):

            if det_data_available and det_view_enable:

                image_current_create_det = copy.copy(image_current);

                det_object_list_new, save_flag = run_create_det_object.create_detection(image_current_create_det, config.label_replace, frame_name = img_file_name, frame_id = frame_index, det_object_list = det_object_list);
                if save_flag:
                    det_object_list = det_object_list_new;

                    print('Saving detections: {}'.format(det_csv_path));
                    DetObject.to_csv(det_csv_path, det_object_list);


        elif key == ord("c"):
            subprocess.call(["subl", "--new-window", agenttype_corr_path]);

        elif key == ord("m"):
            if track_merge_data_available:
                subprocess.call(["subl", "--new-window", config.track_merge]);

        elif key == ord("r"):
            subprocess.call(["subl", "--new-window", traj_reverse_path]);

        elif key == ord("w"):

            if det_view_enable:
                if det_data_available:
                    subprocess.call(["subl", "--new-window", det_csv_path]);

                if det_asso_data_available:
                    subprocess.call(["subl", "--new-window", det_asso_csv_path]);


        elif key == ord("i"):
            subprocess.call(["subl", "--new-window", traj_ingore_path]);


        elif key == ord("t"):

            subprocess.call(["subl", "--new-window", time_ignore_path]);

        elif key == ord("e"):

            # Export trajectorie
            export_trajectories(list_times_ms,
                                traj_list,
                                list_traj_ignore_list=[list_traj_ignore, list_traj_ignore_automatic, list_traj_ignore_timeignore],
                                time_ignore_list=time_ignore_list,
                                det_zone_ignore_list=det_zone_ignore_list,
                                name_sequence=name_sequence,
                                traj_saving_dir=traj_saving_dir,
                                location_name=config.location_name,
                                date=config.date,
                                start_time=config.start_time,
                                delta_ms=config.delta_ms);

        else:
            print_instructions();

if __name__ == '__main__':

    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
