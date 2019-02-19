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

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../'));
sys.path.append(ROOT_DIR);

from box3D_fitting import Box3D_utils

from object_det.mask_rcnn import detect_utils
from tracker import cameramodel as cm
from camera_calib import calib_utils

from utils import cfgutil
from utils.mathutil import *

class Type3DBoxStruct():
    def __init__(self, label, length, width, height):
        self.label = label;
        self.length = length;
        self.width = width;
        self.height = height;
        self.box3D_lwh = [ float(length), float(width), -float(height)];

# Define default 3D box type:
default_type_3DBox_list = [ Type3DBoxStruct('car', 4.0, 2.0, 1.6),\
                            Type3DBoxStruct('bus', 13.0, 3.5, 2.5),\
                            Type3DBoxStruct('person', 0.8, 0.8, 1.9),\
                            Type3DBoxStruct('truck', 13.0, 3.5, 2.5)];

# Create a default cfg file which holds default values for the path
def create_default_cfg():

    config = configparser.ConfigParser();

    config['INPUT_PATH'] = \
                        {'CAMERA_CFG_PATH': '',\
                         'IMAGE_DATA_DIR': '', \
                         'DET_DATA_DIR': '',\
                         'DET_ZONE_F_PATH' :'' }

    config['OUTPUT_PATH'] = \
                        {'BOX3D_DATA_DIR': ''};

    config['OPTIONS'] = \
                        {'IMG_SCALE': '0.7',
                         'SHOW_IMAGES': '0',
                         'SAVE_IMAGES': '1'};


    # Header of the cfg file
    text = '\
##########################################################################################\n\
#\n\
# 3D BOX FITTING OPTIMIZER\n\
#\n\
# Fit a 3D box to the detected object mask\n\
#\n\
# Please modify this config file according to your configuration.\n\
# Path must be ABSOLUTE PATH\n\
##########################################################################################\n\n'

    # Write the cfg file
    with open('OPTIM_3DBOX_MONO.ini', 'w') as configfile:
        configfile.write(text);
        config.write(configfile)

def write_default_type_csv(path_csv, type_3DBox_list):
    csv_open = False;
    with open(path_csv, 'w') as csvfile:

        fieldnames = [];

        # Add new keys
        fieldnames.append('label');
        fieldnames.append('length');
        fieldnames.append('width');
        fieldnames.append('height');

        #Write field name
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
        writer.writeheader();

        for type_3DBox in type_3DBox_list:

            # Field management
            dict_row = {};

            dict_row['label'] = type_3DBox.label;
            dict_row['length'] = type_3DBox.length;
            dict_row['width'] = type_3DBox.width;
            dict_row['height'] = type_3DBox.height;

            # Write detection in CSV
            writer.writerow(dict_row);


def read_type_csv(csv_path):

    data_list = [];

    # Read CSV
    with open(csv_path) as csvfile:
        # Open CSV sa a dict
        reader = csv.DictReader(csvfile)

        # Read each row
        for row in reader:

            fields = row.keys();

            if 'label' in fields:
                label = row['label'];

            if 'width' in fields:
                width = np.float64(row['width']);

            if 'height' in fields:
                height = np.float64(row['height']);

            if 'length' in fields:
                length = np.float64(row['length']);

            # Create data struct
            type_data  = Type3DBoxStruct(label, length, width, height);
            data_list.append(type_data);

    return data_list

# Write the box detected in csv, one csv per frame
def write_3D_box_csv(path_csv, box_3D_results):

    with open(path_csv, 'w') as csvfile:

        # Write header
        fieldnames = [];

        # Add new keys
        fieldnames.append('det_id');
        fieldnames.append('percent_overlap');
        fieldnames.append('box_3D_phi');
        fieldnames.append('box_3D_x');
        fieldnames.append('box_3D_y');
        fieldnames.append('box_3D_z');
        fieldnames.append('box_3D_l');
        fieldnames.append('box_3D_w');
        fieldnames.append('box_3D_h');

        #Write field name
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
        writer.writeheader();

        for index in range(0,len(box_3D_results)):

            # Field management
            dict_row = {};

            dict_row['det_id'] = box_3D_results[index]['det_id'];
            dict_row['percent_overlap'] = box_3D_results[index]['percent_overlap'];

            # Raw box:
            dict_row['box_3D_phi'] = box_3D_results[index]['box_3D'][0];
            dict_row['box_3D_x'] = box_3D_results[index]['box_3D'][1];
            dict_row['box_3D_y'] = box_3D_results[index]['box_3D'][2];
            dict_row['box_3D_z'] = box_3D_results[index]['box_3D'][3];
            dict_row['box_3D_l'] = box_3D_results[index]['box_3D'][4];
            dict_row['box_3D_w'] = box_3D_results[index]['box_3D'][5];
            dict_row['box_3D_h'] = box_3D_results[index]['box_3D'][6];


            # Write detection in CSV
            writer.writerow(dict_row);
def main():

    # Print instructions
    print("############################################################")
    print("3D box fitting optimizer")
    print("############################################################\n")

    # ##########################################################
    # # Parse Arguments
    # ##########################################################
    argparser = argparse.ArgumentParser(
        description='3D box fitting optimizer')
    argparser.add_argument(
        '-i', '--init',
        action='store_true',
        help='Generate config files to fill in order to run the 3D box optimization: OPTIM_3DBOX_MONO.ini and optim_3Dbox_mono_type.csv')
    argparser.add_argument(
        '-cfg', '--config',
        default='OPTIM_3DBOX_MONO.ini',
        help='Path to the config file')
    args = argparser.parse_args();


    # In init mode
    if args.init:

        # Create csv templates
        write_default_type_csv('optim_3Dbox_mono_type.csv', default_type_3DBox_list);

        # Create default config file
        create_default_cfg();

        print('Please fill the config files and restart the program:\n-OPTIM_3DBOX_MONO.ini\n-optim_3Dbox_mono_type.csv')

        return;

    # ##########################################################
    # # Read config file:
    # ##########################################################
    config = cfgutil.read_cfg(args.config);

    # Read Type Struct from csv
    try:
        type_3DBox_list = read_type_csv('optim_3Dbox_mono_type.csv');
    except Exception as e:
        print('[ERROR]: Error opening optim_3Dbox_mono_type.csv {}'.format(e))
        return;

    # Create output dorectory to save filtered image:
    os.makedirs(config['OUTPUT_PATH']['BOX3D_DATA_DIR'], exist_ok=True)
    box3d_data_csv_dir = os.path.join(config['OUTPUT_PATH']['BOX3D_DATA_DIR'], 'csv');
    os.makedirs(box3d_data_csv_dir, exist_ok=True)
    box3d_data_img_dir = os.path.join(config['OUTPUT_PATH']['BOX3D_DATA_DIR'], 'img');
    os.makedirs(box3d_data_img_dir, exist_ok=True)

    # Save the cfg file with the output:
    try:
        cfg_save_path = args.config.split('/')[-1];
        cfg_save_path = os.path.join(config['OUTPUT_PATH']['BOX3D_DATA_DIR'], cfg_save_path);
        copyfile(args.config, cfg_save_path)
    except Exception as e:
        print('[ERROR]: Error saving config file in output folder:\n')
        print('{}'.format(e))
        return;

    # Option for output
    SHOW_IMAGES = bool(float(config['OPTIONS']['SHOW_IMAGES']))
    SAVE_IMAGES = bool(float(config['OPTIONS']['SAVE_IMAGES']))

    ##########################################################
    # Camera Parameters
    ##########################################################

    cam_model_1 = calib_utils.read_camera_calibration(config['INPUT_PATH']['CAMERA_CFG_PATH']);
    cam_scale_factor = float(config['OPTIONS']['IMG_SCALE']);
    if cam_scale_factor < 0:
        print('[ERROR]: Image scale factor < 0: {}'.format(cam_scale_factor))
    cam_model_1.apply_scale_factor(cam_scale_factor,cam_scale_factor);

    ##########################################################
    # Images Folder:
    ##########################################################

    # Load images
    list_file = os.listdir(config['INPUT_PATH']['IMAGE_DATA_DIR']);
    list_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    ##########################################################
    # Detection zone
    ##########################################################

    # Detection Zone YML file:
    pt_det_zone_FNED = None;
    pt_det_zone_pix = None;
    if config['INPUT_PATH']['DET_ZONE_F_PATH'] != '':
        fs_read = cv2.FileStorage(config['INPUT_PATH']['DET_ZONE_F_PATH'], cv2.FILE_STORAGE_READ)
        pt_det_zone_FNED = fs_read.getNode('model_points_FNED').mat();

        pt_det_zone_pix = Box3D_utils.pt_det_zone_FNED_to_pix(pt_det_zone_FNED, cam_model_1)

    ##########################################################
    # MultiThread Management: Using Pool is very straightforward
    ##########################################################

    # Create pool of thead
    pool = ThreadPool(50);
    total_time = time.time()

    # Loop trough tracker
    for image_name in list_file:
        start_time = time.time()
        print('Image: {}'.format(image_name))

        # CSV name management
        csv_name = image_name.split('.')[0] + '_det.csv';

        try:
            data_det = detect_utils.read_dict_csv(os.path.join(config['INPUT_PATH']['DET_DATA_DIR'], csv_name));
        except Exception as e:
            print(e);
            print("Could not open detection csv {}".format(os.path.join(config['INPUT_PATH']['DET_DATA_DIR'], csv_name)));
            break;

        # Nober of detections
        nb_det = len(data_det);

        # Open Image
        im_1 = cv2.imread(os.path.join(config['INPUT_PATH']['IMAGE_DATA_DIR'], image_name));

        # Scale image:
        im_1 = cv2.resize(im_1,None,fx=float(config['OPTIONS']['IMG_SCALE']), fy=float(config['OPTIONS']['IMG_SCALE']), interpolation = cv2.INTER_CUBIC)

        im_current_1 = copy.copy(im_1);
        im_size_1 = (im_1.shape[0], im_1.shape[1]);

        # Construct array of input so that each worker can work on one input:
        # https://www.codementor.io/lance/simple-parallelism-in-python-du107klle
        array_inputs = [];
        for det_ind in range(0, nb_det):

            # Get the detections from Mask-RCNN csv:
            # Contains: calss_id, mask, etc...
            det_dict = data_det[det_ind];

            for type_3DBox in type_3DBox_list:

                if det_dict['label'] ==  type_3DBox.label:

                    input_dict ={};

                    mask = det_dict['mask'];
                    mask = Box3D_utils.scale_mask(float(config['OPTIONS']['IMG_SCALE']), float(config['OPTIONS']['IMG_SCALE']), mask);
                    input_dict['mask'] = mask;


                    roi = det_dict['roi'];
                    roi = Box3D_utils.scale_roi(float(config['OPTIONS']['IMG_SCALE']), float(config['OPTIONS']['IMG_SCALE']), roi);
                    input_dict['roi'] = roi;

                    input_dict['det_id'] = det_dict['det_id'];

                    input_dict['cam_model'] = cam_model_1;
                    input_dict['im_size'] =  im_size_1;
                    input_dict['box_size'] = type_3DBox.box3D_lwh;

                    if not (pt_det_zone_pix is None):
                        in_zone = Box3D_utils.in_detection_zone(roi, pt_det_zone_pix)
                        if in_zone:
                            array_inputs.append(input_dict);
                    else:
                        array_inputs.append(input_dict);


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

        # Save results in CSV
        csv_name = image_name.split('.')[0] + "_3Dbox.csv"
        path_csv = os.path.join(box3d_data_csv_dir, csv_name)
        write_3D_box_csv(path_csv, results);

        # Draw results on the Image:
        for result in results:

            # # Do not plot the 3D box if overlap < 70 %
            # if result['percent_overlap'] < 0.7:
            #     continue;

            param_box = result['box_3D'];
            mask_1 = result['mask'];

            list_pt_F = Box3D_utils.create_3Dbox(param_box);

            im_current_1 = Box3D_utils.draw_3Dbox(im_current_1, cam_model_1, list_pt_F);

            # Mask box
            mask_box_1 = Box3D_utils.create_mask(im_size_1, cam_model_1, list_pt_F);

            o_1, mo_1, mo_1_b = Box3D_utils.overlap_mask(mask_1, mask_box_1);

            print("Overlap total: {}".format(o_1));

            # Do not plot the 3D box if overlap < 70 %
            if result['percent_overlap'] < 0.5:
                im_current_1 = Box3D_utils.draw_mask(im_current_1, mask_box_1, (255,0,0));
                im_current_1 = Box3D_utils.draw_mask(im_current_1, mask_1, (255,0,0));

            else:
                im_current_1 = Box3D_utils.draw_mask(im_current_1, mask_box_1, (0,0,255));
                im_current_1 = Box3D_utils.draw_mask(im_current_1, mask_1, (0,255,255));

            # im_current_1 = Box3D_utils.draw_boundingbox(im_current_1, r_1);

        if SHOW_IMAGES:
            cv2.imshow("Camera 1", im_current_1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save the Image
        if SAVE_IMAGES:
            cv2.imwrite( os.path.join(box3d_data_img_dir, image_name), im_current_1 );
            print('\n ===> Execution Time', round((time.time() - start_time), 5 ), '\n' )

    print('\n ===> Total execution time', total_time)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    # except Exception as e:
    #     print('[Error]: {}'.format(e))