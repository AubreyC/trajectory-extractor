# -*- coding: utf-8 -*-

##########################################################################################
#
# MEASUREMENT ASSOCIATION TO FORM TRACKS
#
# Association is done based on Intersection-Over-Union between masks of successive frames
#
##########################################################################################


import os
import sys

import random
import math
import numpy as np
import time

import csv
import configparser
import argparse
from shutil import copyfile

import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../'));
sys.path.append(ROOT_DIR);

from det_association import overlapTrack as tk_over
from det_association import multipleOverlapAssociation
from object_det.mask_rcnn import detect_utils
from utils import cfgutil

# Create a default cfg file which holds default values for the path
def create_default_cfg():

    config = configparser.ConfigParser();

    config['INPUT_PATH'] = \
                        {'IMAGE_DATA_DIR': '', \
                         'DET_DATA_DIR': ''}

    config['OUTPUT_PATH'] = \
                        {'OUTPUT_DIR': ''}

    config['OPTIONS'] = \
                        {'ASSOCIATE_WITH_LABEL': '0', \
                         'THRESHOLD_OVERLAP': '0.3', \
                         'NB_FRAME_PAST': '10', \
                         'SHOW_IMAGES': '0', \
                         'SAVE_IMAGES': '1',\
                         'SAVE_CSV': '1'};

    # Header of the cfg file
    text = '#\n# Measurement association to form tracks based on mask overlap between consecutive detected objects:\n# Please modify this config file according to your configuration.\n# Path must bo ABSOLUTE PATH\n#\n\n'
    text = '\
##########################################################################################\n\
#\n\
# MEASUREMENT ASSOCIATION TO FORM TRACKS\n\
#\n\
# Association is done based on Intersection-Over-Union between masks of successive frames\n\
#\n\
# Please modify this config file according to your configuration.\n\
# Path must be ABSOLUTE PATH\n\
##########################################################################################\n\n'
    # Write the cfg file
    with open('DET_ASSOCIATION_CFG.ini', 'w') as configfile:
        configfile.write(text);
        config.write(configfile)


def create_tracker_dict(r, class_names, det_ind, track_ind):

    #Create dict for the detection
    dict_track = detect_utils.create_det_dict(r, class_names, det_ind);

    # Add Track id:
    dict_track['track_id'] = track_ind;

    return dict_track

def write_track_csv(path_csv, r, class_names, track_numbers):

    csv_open = False;
    with open(path_csv, 'w') as csvfile:

        fieldnames = [];

        # Add new keys
        fieldnames.append('det_id');
        fieldnames.append('track_id');

        #Write field name
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
        writer.writeheader();

        for index in range(0,len(r['rois'])):

            # Very simple overlap csv
            dict_track = {};
            dict_track['det_id'] = index;
            dict_track['track_id'] = track_numbers[index];

            # Write detection in CSV
            writer.writerow(dict_track);


def main():

  # Print instructions
    print("############################################################")
    print("Measurement Association based on Intersection-Over-Union")
    print("############################################################\n")

    # ##########################################################
    # # Parse Arguments
    # ##########################################################
    argparser = argparse.ArgumentParser(
        description='Measurement association')
    argparser.add_argument(
        '-i', '--init',
        action='store_true',
        help='Generate config files to fill in order to run the measurement accosiation: DET_ASSOCIATION_CFG.ini')
    argparser.add_argument(
        '-cfg', '--config',
        default='DET_ASSOCIATION_CFG.ini',
        help='Path to the config file')
    args = argparser.parse_args();

  # In init mode
    if args.init:

        # Create default config file
        create_default_cfg();

        print('Please fill the config files and restart the program:\n-DET_ASSOCIATION_CFG.ini')
        return;

    # ##########################################################
    # # Read config file:
    # ##########################################################

    config = cfgutil.read_cfg(args.config);


    # Create output dorectory to save filtered image:
    os.makedirs(config['OUTPUT_PATH']['OUTPUT_DIR'], exist_ok=True);
    overlap_img_dir = os.path.join(config['OUTPUT_PATH']['OUTPUT_DIR'], 'img');
    os.makedirs(overlap_img_dir, exist_ok=True)
    overlap_csv_dir = os.path.join(config['OUTPUT_PATH']['OUTPUT_DIR'], 'csv');
    os.makedirs(overlap_csv_dir, exist_ok=True)

    # Save the cfg file with the output:
    try:
        cfg_save_path = args.config.split('/')[-1];
        cfg_save_path = os.path.join(config['OUTPUT_PATH']['OUTPUT_DIR'], cfg_save_path);
        copyfile(args.config, cfg_save_path)
    except Exception as e:
        print('[ERROR]: Error saving config file in output folder:\n')
        print('{}'.format(e))
        return;

    # Option for output
    show_images = bool(float(config['OPTIONS']['SHOW_IMAGES']));
    save_images = bool(float(config['OPTIONS']['SAVE_IMAGES']));
    save_csv = bool(float(config['OPTIONS']['SAVE_CSV']));

    associate_with_label = bool(float(config['OPTIONS']['ASSOCIATE_WITH_LABEL']));

    # Threshold to associate measurement: Need to overlap by at least this threshold
    threshold_overlap = float(config['OPTIONS']['THRESHOLD_OVERLAP']);

    # Number of frame we look into the past to associate measurement to a track
    nb_frame_past = int(config['OPTIONS']['NB_FRAME_PAST']);

    # Print options
    print("Options: SHOW_IMAGES: {}, SAVE_IMAGES: {}, SAVE_CSV: {}".format(show_images, save_images, save_csv));

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']



    # Load images
    list_file = os.listdir(config['INPUT_PATH']['IMAGE_DATA_DIR']);
    list_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # Construct tracker
    tk_overlap = multipleOverlapAssociation.MultipleOverlapAssociation(associate_with_label, threshold_overlap, nb_frame_past);

    for image_name in list_file:
        print('Image: {}'.format(image_name))
        start_time = time.time()


        # CSV name management
        csv_name = image_name.split('.')[0] + '_det.csv';

        # Read and convert the mask:
        r = detect_utils.read_detection_csv(os.path.join(config['INPUT_PATH']['DET_DATA_DIR'], csv_name), class_names, only_mask_cont=False);

        if r is None:
            continue;

        track_numbers, colors = tk_overlap.push_detection(r);


        # Add annotation on Image:
        if save_images or show_images:
            cv_image = cv2.imread(os.path.join(config['INPUT_PATH']['IMAGE_DATA_DIR'], image_name));
            d = np.zeros(cv_image.shape, dtype = cv_image.dtype)

            nb_det = len(r['rois']);
            for i in range(0, nb_det):
                x_1 = int(r['rois'][i][1]);
                y_1 = int(r['rois'][i][0]);
                x_2 = int(r['rois'][i][3]);
                y_2 = int(r['rois'][i][2]);

                tl = (x_1, y_1)
                br = (x_2, y_2)
                # label = result['label']
                # confidence = result['confidence']
                text = 'T: {}, D: {}, C: {}'.format(track_numbers[i], i, class_names[r['class_ids'][i]])
                cv_image = cv2.rectangle(cv_image, tl, br, (255*colors[i][0], 255*colors[i][1], 255*colors[i][2]), 1)
                cv_image = cv2.putText(cv_image, text, tl, cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

                clrImg = np.zeros(cv_image.shape, cv_image.dtype)
                clrImg[:,:] = (255*colors[i][0], 255*colors[i][1], 255*colors[i][2])

                m = np.array(r['masks'][:,:,i], dtype = "uint8")
                clrMask = cv2.bitwise_and(clrImg, clrImg, mask=m);

                cv2.addWeighted(cv_image, 1.0, clrMask, 0.5, 0.0, cv_image)

            if show_images:
                cv2.imshow('frame', cv_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_images:
                cv2.imwrite( os.path.join(overlap_img_dir, image_name), cv_image );

        if save_csv:
            name_csv = image_name.split('.')[0] + '_detassociation.csv';
            write_track_csv(os.path.join(overlap_csv_dir, name_csv), r, class_names, track_numbers);

        print('\n ===> Execution Time', round((time.time() - start_time), 5 ), '\n' )


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')