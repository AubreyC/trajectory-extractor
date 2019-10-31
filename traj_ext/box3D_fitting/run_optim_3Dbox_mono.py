# -*- coding: utf-8 -*-

##########################################################################################
#
# 3D BOX FITTING OPTIMIZER
#
# Fit a 3D box to the detected object mask
#
##########################################################################################

import numpy as np
import time
import cv2
import copy
from scipy.optimize import linear_sum_assignment
import os
import sys
import csv
import threading
import argparse
import scipy.optimize as opt
from multiprocessing.dummy import Pool as ThreadPool
import configparser
from shutil import copyfile
import json

from traj_ext.box3D_fitting import Box3D_utils

from traj_ext.object_det.mask_rcnn import detect_utils
from traj_ext.tracker import cameramodel

from traj_ext.utils import cfgutil
from traj_ext.utils.mathutil import *

from traj_ext.object_det import det_object
from traj_ext.utils import det_zone
from traj_ext.box3D_fitting import box3D_object

def main(args_input):

    # Print instructions
    print("############################################################")
    print("Object Detector with Mask-RCNN")
    print("############################################################\n")

    # ##########################################################
    # # Parse Arguments
    # ##########################################################
    argparser = argparse.ArgumentParser(
        description='Object Detector with Mask-RCNN')
    argparser.add_argument(
        '-image_dir',
        default='',
        help='Path of the image folder')
    argparser.add_argument(
        '-det_dir',
        default='',
        help='Path of the detection folder');
    argparser.add_argument(
        '-det_zone_fned',
        default='',
        help='Path of the detection zone FNED file');
    argparser.add_argument(
        '-type_box3D',
        default='',
        help='Path of box3D type csv file');
    argparser.add_argument(
        '-camera_model',
        default='',
        help='Path to the camera model yaml');

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
        '-img_scale',
        type = float,
        default = 0.7,
        help='Image scaling factor to improve speed');

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

    return run_optim_3Dbox_mono(args);

def run_optim_3Dbox_mono(config):

    # Create output folder
    output_dir = config.output_dir;
    output_dir = os.path.join(output_dir, 'box3D');
    os.makedirs(output_dir, exist_ok=True)

    # Save the cfg file with the output:
    try:
        cfg_save_path = os.path.join(output_dir, 'optim_box3D_mono.json');
        with open(cfg_save_path, 'w') as json_file:
            config_dict = vars(config);
            json.dump(config_dict, json_file, indent=4)
    except Exception as e:
        print('[ERROR]: Error saving config file in output folder:\n')
        print('{}'.format(e))
        return False;

    # Type of Box 3D:
    type_3DBox_list = box3D_object.Type3DBoxStruct.default_3DBox_list();
    if os.path.isfile(config.type_box3D):
        type_3DBox_list = box3D_object.Type3DBoxStruct.read_type_csv(config.type_box3D);

    box3D_save_path = os.path.join(output_dir, 'box3D_type.csv');
    box3D_object.Type3DBoxStruct.write_box3D_type_csv(box3D_save_path, type_3DBox_list);

    # Create output dorectory to save filtered image:
    box3d_data_csv_dir = os.path.join(output_dir, 'csv');
    os.makedirs(box3d_data_csv_dir, exist_ok=True)
    box3d_data_img_dir = os.path.join(output_dir, 'img');
    os.makedirs(box3d_data_img_dir, exist_ok=True)

    # Option for output
    save_csv = not config.no_save_csv;
    save_images = not config.no_save_images;
    show_images = config.show_images
    frame_limit = config.frame_limit;

    ##########################################################
    # Camera Parameters
    ##########################################################

    cam_model_1 = cameramodel.CameraModel.read_from_yml(config.camera_model);
    cam_scale_factor = config.img_scale;
    if cam_scale_factor < 0:
        print('[ERROR]: Image scale factor < 0: {}'.format(cam_scale_factor))
    cam_model_1.apply_scale_factor(cam_scale_factor,cam_scale_factor);

    ##########################################################
    # Images Folder:
    ##########################################################

    # Load images
    list_img_file = os.listdir(config.image_dir);
    list_img_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # Limit images with frame_limit
    if frame_limit > 0:
        list_img_file = list_img_file[:frame_limit];

    # ##########################################################
    # # Detection zone
    # ##########################################################

    # det_zone_IM = None;
    # if os.path.isfile(config.det_zone_fned):
    #     det_zone_FNED = det_zone.DetZoneFNED.read_from_yml(config.det_zone_fned);
    #     det_zone_IM = det_zone_FNED.create_det_zone_image(cam_model_1);

    ##########################################################
    # MultiThread Management: Using Pool is very straightforward
    ##########################################################

    # Create pool of thead
    pool = ThreadPool(50);
    total_time = time.time()

    # Loop trough tracker
    for current_index, image_name in enumerate(list_img_file):

        start_time = time.time()
        print('Image: {}'.format(image_name))

        # CSV name management
        csv_name = image_name.split('.')[0] + '_det.csv';

        # Open detections
        det_object_list = det_object.DetObject.from_csv(os.path.join(config.det_dir, csv_name), expand_mask = True);

        # Nober of detections
        nb_det = len(det_object_list);

        # Open Image
        im_1 = cv2.imread(os.path.join(config.image_dir, image_name));

        # Scale image:
        im_1 = cv2.resize(im_1,None,fx=config.img_scale, fy=config.img_scale, interpolation = cv2.INTER_CUBIC)

        im_current_1 = copy.copy(im_1);
        im_size_1 = (im_1.shape[0], im_1.shape[1]);

        # Construct array of input so that each worker can work on one input:
        # https://www.codementor.io/lance/simple-parallelism-in-python-du107klle
        array_inputs = [];
        for det_ind in range(0, nb_det):

            # Get the detections from Mask-RCNN csv:
            # Contains: calss_id, mask, etc...
            det = det_object_list[det_ind];

            for type_3DBox in type_3DBox_list:

                if det.label ==  type_3DBox.label:

                    det_scaled = det.to_scale(config.img_scale, config.img_scale);

                    input_dict = {};

                    input_dict['mask'] = det_scaled.det_mask;

                    input_dict['roi'] = det_scaled.det_2Dbox;

                    input_dict['det_id'] = det_scaled.det_id;

                    input_dict['cam_model'] = cam_model_1;
                    input_dict['im_size'] =  im_size_1;
                    input_dict['box_size'] = type_3DBox.box3D_lwh;

                    array_inputs.append(input_dict);

                    # # If det_zone_IM is defined:
                    # if not (det_zone_IM is None):

                    #     # Only add it the the processing queue if its in zone
                    #     if det_zone_IM.in_zone(det_scaled.get_center_det_2Dbox()):
                    #         array_inputs.append(input_dict);

                    # else:
                    #     array_inputs.append(input_dict);


        # Run the array of inputs through the pool of workers
        print("Thread alive: {}".format(threading.active_count()))
        not_done = True;
        while(not_done):
            try:
                results = pool.map(Box3D_utils.find_3Dbox_multithread, array_inputs);
                not_done = False;
            except Exception as e:
                print(e);
                not_done = True;


        box3D_list = [];
        for result in results:
            box3D_list.append(result['box_3D']);

        # Save results in CSV
        if save_csv:
            csv_name = image_name.split('.')[0] + "_3Dbox.csv"
            path_csv = os.path.join(box3d_data_csv_dir, csv_name)
            box3D_object.Box3DObject.to_csv(path_csv, box3D_list);

        # Draw results on the Image:
        for result in results:

            # # Do not plot the 3D box if overlap < 70 %
            # if result['percent_overlap'] < 0.7:
            #     continue;

            # Get result:
            box3D_result = result['box_3D'];
            mask_1 = result['mask'];

            # Display box on image
            box3D_result.display_on_image(im_current_1, cam_model_1);

            mask_box_1 = box3D_result.create_mask(cam_model_1, im_size_1);

            o_1, mo_1, mo_1_b = Box3D_utils.overlap_mask(mask_1, mask_box_1);
            print("Overlap total: {}".format(o_1));

            # Do not plot the 3D box if overlap < 70 %
            if box3D_result.percent_overlap < 0.5:
                im_current_1 = det_object.draw_mask(im_current_1, mask_box_1, (255,0,0));
                im_current_1 = det_object.draw_mask(im_current_1, mask_1, (255,0,0));

            else:
                im_current_1 = det_object.draw_mask(im_current_1, mask_box_1, (0,0,255));
                im_current_1 = det_object.draw_mask(im_current_1, mask_1, (0,255,255));

            # im_current_1 = Box3D_utils.draw_boundingbox(im_current_1, r_1);

        if show_images:
            cv2.imshow("Camera 1", im_current_1)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        # Save the Image
        if save_images:
            cv2.imwrite( os.path.join(box3d_data_img_dir, image_name), im_current_1 );
            print('\n ===> Execution Time', round((time.time() - start_time), 5 ), '\n' )

    print('\n ===> Total execution time', total_time)
    cv2.destroyAllWindows()

    return True;

if __name__ == '__main__':

    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')