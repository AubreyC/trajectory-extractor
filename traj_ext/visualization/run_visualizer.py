# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-06-27 13:58:53
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-27 22:29:12

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

from traj_ext.visualization import run_inspect_traj
from traj_ext.hd_map.HD_map import HDmap


from traj_ext.utils import det_zone
from traj_ext.utils import mathutil

from traj_ext.tracker import EKF_utils


def main(args_input):

    # Print instructions
    print("############################################################")
    print("Visualize the final trajectories")
    print("############################################################\n")

    # ##########################################################
    # # Parse Arguments
    # ##########################################################
    argparser = argparse.ArgumentParser(
        description='Visualize the final trajectories')
    argparser.add_argument(
        '-traj',
        default='',
        help='Path of the trajectories csv')
    argparser.add_argument(
        '-traj_person',
        default='',
        help='Path of the person trajectories csv')
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
        '-hd_map',
        type = str,
        default = '',
        help='Path to the HD map');
    argparser.add_argument(
        '-no_label',
        action ='store_true',
        help='Do not display track id');

    argparser.add_argument(
        '-output_dir',
        type=str,
        default='',
        help='Path of the output');

    argparser.add_argument(
        '-export',
        type=bool,
        default = False,
        help='Export trajectories directly and exit the program (for automated process)');

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

    run_visualize_traj(args);


def run_visualize_traj(config):

    # Create output folder
    output_dir = config.output_dir;
    output_dir = os.path.join(output_dir, 'visualizer');
    os.makedirs(output_dir, exist_ok=True)

    # Create output sub-folder
    image_raw_saving_dir = os.path.join(output_dir, 'img_raw');
    image_annoted_saving_dir = os.path.join(output_dir, 'img_annoted');
    image_hd_map_saving_dir = os.path.join(output_dir, 'img_hdmap');
    image_concat_saving_dir = os.path.join(output_dir, 'img_concat');

    os.makedirs(image_raw_saving_dir, exist_ok=True)
    os.makedirs(image_annoted_saving_dir, exist_ok=True)
    os.makedirs(image_hd_map_saving_dir, exist_ok=True)
    os.makedirs(image_concat_saving_dir, exist_ok=True)

    # Save the cfg file with the output:
    try:
        cfg_save_path = os.path.join(output_dir, 'visualize_traj_cfg.json');
        with open(cfg_save_path, 'w') as json_file:
            config_dict = vars(config);
            json.dump(config_dict, json_file, indent=4)
    except Exception as e:
        print('[ERROR]: Error saving config file in output folder:\n')
        print('{}'.format(e))
        return;

    #Person trajectories
    traj_person_list = [];
    traj_person_available = os.path.isfile(config.traj_person);
    if traj_person_available:
        traj_person_list = Trajectory.read_trajectory_panda_csv(config.traj_person);

    # Open objects
    cam_model = CameraModel.read_from_yml(config.camera_street);
    det_zone_FNED = det_zone.DetZoneFNED.read_from_yml(config.det_zone_fned);

    # Time
    list_times_ms_path = config.time;
    if list_times_ms_path == '':
        list_times_ms_path = config.traj.replace('traj.csv', 'time_traj.csv');
    if not os.path.isfile(list_times_ms_path):
        print('[ERROR]: Traj time file not found: {}'.format(list_times_ms_path))
        return;
    list_times_ms = trajutil.read_time_list_csv(list_times_ms_path);

    # Trajectories
    traj_list = Trajectory.read_trajectory_panda_csv(config.traj);

    sat_view_available = False;
    sat_view_enable = False;
    if os.path.isfile(config.camera_sat) and os.path.isfile(config.camera_sat_img):
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

    hd_map_available = os.path.isfile(config.hd_map);
    if hd_map_available:

        hd_map = HDmap.from_csv(config.hd_map);

        cam_model_hdmap, image_hdmap = hd_map.create_view();

        image_hdmap = hd_map.display_on_image(image_hdmap, cam_model_hdmap);


    # Shrink det zone for complete
    det_zone_FNED_complete = None;
    if config.shrink_zone < 1:
        det_zone_FNED_complete = det_zone_FNED.shrink_zone(config.shrink_zone);
        for traj in traj_list:
            if not traj.check_is_complete(det_zone_FNED_complete):
                print('Track: {} time_ms: {} not complete'.format(traj.get_id(), traj.get_start_trajoint().time_ms))



    skip_value = 1;
    frame_index = 0;
    export_mode = False;

    # If export mode is direclty asked
    if config.export:
        export_mode = True;

    while True:

        #######################################################
        ## Show Trajectories
        #######################################################

        # Get Time ms
        frame_index = mathutil.clip(frame_index, 0, len(list_times_ms)-1);
        time_ms = list_times_ms[frame_index];

        if image_in_dir:
            img_file_name = list_img_file[frame_index];
            image_current = cv2.imread(os.path.join(config.image_dir, img_file_name));
        else:
            # Copy image
            image_current = image;


        if not (image_current is None):

            print('Showing: frame_id: {} image: {}'.format(frame_index, img_file_name));

            # Display traj
            image_current_traj, _ = run_inspect_traj.display_traj_on_image(time_ms, cam_model, image_current, traj_list, det_zone_FNED_list = [det_zone_FNED], no_label = config.no_label);
            image_current_traj, _ = run_inspect_traj.display_traj_on_image(time_ms, cam_model, image_current_traj, traj_person_list, det_zone_FNED_list = [det_zone_FNED], no_label = config.no_label);

            # Show image
            cv2.imshow('Trajectory visualizer', image_current_traj)

            if sat_view_available:

                if sat_view_enable:

                    # Display traj
                    image_sat_current, _ = run_inspect_traj.display_traj_on_image(time_ms, cam_model_sat, image_sat, traj_list, det_zone_FNED_list = [det_zone_FNED], no_label = config.no_label);
                    image_sat_current, _ = run_inspect_traj.display_traj_on_image(time_ms, cam_model_sat, image_sat_current, traj_person_list, det_zone_FNED_list = [det_zone_FNED], no_label = config.no_label)
                    # Show image
                    cv2.imshow('Sat View merged', image_sat_current)

            if hd_map_available:

                # Display traj
                image_hdmap_current, _ = run_inspect_traj.display_traj_on_image(time_ms, cam_model_hdmap, image_hdmap, traj_list, det_zone_FNED_list = [det_zone_FNED], no_label = config.no_label, velocity_label=True);
                image_hdmap_current, _ = run_inspect_traj.display_traj_on_image(time_ms, cam_model_hdmap, image_hdmap_current, traj_person_list, det_zone_FNED_list = [det_zone_FNED], no_label = config.no_label, velocity_label=True);

                # Show image
                cv2.imshow('HD map view merged', image_hdmap_current)


                image_concat = EKF_utils.concatenate_images(image_current_traj, image_hdmap_current)
                cv2.imshow('View: Camera and HD map', image_concat)

            if export_mode:

                img_annoted_name = img_file_name.split('.')[0] + '_annotated.png';
                print('Saving: frame_id: {} image: {}'.format(frame_index, img_annoted_name))

                image_annoted_path = os.path.join(image_annoted_saving_dir, img_annoted_name);
                cv2.imwrite(image_annoted_path, image_current_traj);

                print('Saving: frame_id: {} image: {}'.format(frame_index, img_file_name))
                image_raw_path = os.path.join(image_raw_saving_dir, img_file_name);
                cv2.imwrite(image_raw_path, image_current);

                if hd_map_available:

                    img_hdmap_name = img_file_name.split('.')[0] + '_hdmap.png';
                    print('Saving: frame_id: {} image: {}'.format(frame_index, img_hdmap_name))
                    image_hdmap_path = os.path.join(image_hd_map_saving_dir, img_hdmap_name);
                    cv2.imwrite(image_hdmap_path, image_hdmap_current);

                    img_concat_name = img_file_name.split('.')[0] + '_concat.png';
                    print('Saving: frame_id: {} image: {}'.format(frame_index, img_concat_name))
                    image_concat_path = os.path.join(image_concat_saving_dir, img_concat_name);
                    cv2.imwrite(image_concat_path, image_concat);

        else:
            print('Not found: frame_id: {} image: {}'.format(frame_index, img_file_name));

        #######################################################
        ## Control keys
        #######################################################

        if frame_index == (len(list_times_ms)-1):
            print('Export view: Done');
            export_mode = False;

            # Exit when it is done if in export directly mode
            if config.export:
                break;

        wait_key = 0;
        if export_mode:
            frame_index +=1;
            wait_key = 1;

        key = cv2.waitKey(wait_key) & 0xFF

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

        # Escape: Quit the program
        elif key == 27:
            break;

        # Escape: Quit the program
        elif key == ord('z'):
            if sat_view_available:
                if sat_view_enable:
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

        elif key == ord("e"):

            if not export_mode:
                export_mode = True;
                frame_index = 0;
                print('Mode: Export mode started');

            else:
                export_mode = False;
                print('Mode: Export mode stopped');

        elif key == 255:
            pass;

        else:

            print('\nInstruction:\n- n: Next frame\
                                 \n- b: Jump 1000 frame forward\
                                 \n- p: Previous frame\
                                 \n- b: Jump 1000 frame backward\
                                 \n- +: Increase skip value\
                                 \n- -: Decrease skip value\
                                 \n- d: Open detection window\
                                 \n- c: Open Agent type correction\
                                 \n- m: Open merging file with sublime text\
                                 \n- s: Enable saving form current frame\
                                 \n- Click on Detection window: Enable/disable detections\
                                 \n- f: Display only complete trajectories\
                                 \n- i: Open ignore trajectory file\
                                 \n- e: Export trajectory file\
                                 \n- esc: Quit\
                                 \n')


if __name__ == '__main__':

    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')