# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-02-08 15:52:10
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

###################################################################################
#
# run_manual_calib.py little script that aims at finding a camera calibration parameters
# from GPS location and associated pixel location on the image. GPS coordinates can
# be found using OpenStreetMap, Google Earth or other servives.
#
###################################################################################

# Inspired from: https://www.learnopencv.com/tag/solvepnp/

#!/usr/bin/env python

import cv2
import numpy as np
import sys
import os
import scipy.optimize as opt
import yaml
import argparse
import configparser
import csv

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../'));
sys.path.append(ROOT_DIR);

from camera_calib import calib_utils
from utils import cfgutil
from utils import mathutil

# Create a default cfg file which holds default values for the path
def create_default_cfg():

    config = configparser.ConfigParser();

    config['INPUT_PATH'] = \
                        {'CAMERA_IMG_PATH': '',\
                         'CALIB_POINT_PATH': ''}

    config['OPTIONS'] = \
                        {'SATELLITE_MODE': '0'};

    # Header of the cfg file
    text = '\
##########################################################################################\n\
#\n\
# MANUAL CAMERA CALIBRATION\n\
#\n\
# Please modify this config file according to your configuration.\n\
# Path must be ABSOLUTE PATH\n\
##########################################################################################\n\n'

    # Write the cfg file
    with open('CAMERA_CALIB_MANUAL_CFG.ini', 'w') as configfile:
        configfile.write(text);
        config.write(configfile)

def write_default_latlon_csv(path_csv):
    csv_open = False;
    with open(path_csv, 'w') as csvfile:

        fieldnames = [];

        # Add new keys
        fieldnames.append('pixel_x');
        fieldnames.append('pixel_y');
        fieldnames.append('lat_deg');
        fieldnames.append('lon_deg');
        fieldnames.append('origin_lat_deg');
        fieldnames.append('origin_lon_deg');

        #Write field name
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
        writer.writeheader();


def write_default_cartesian_csv(path_csv):
    csv_open = False;
    with open(path_csv, 'w') as csvfile:

        fieldnames = [];

        # Add new keys
        fieldnames.append('pixel_x');
        fieldnames.append('pixel_y');
        fieldnames.append('pos_x');
        fieldnames.append('pos_y');

        #Write field name
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
        writer.writeheader();

def read_csv(csv_path):

    data_list = [];

    # Create dict
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:

            fields = row.keys();
            item = {};

            if 'pixel_x' in fields:
                item['pixel_x'] = int(row['pixel_x']);

            if 'pixel_y' in fields:
                item['pixel_y'] = int(row['pixel_y']);

            if 'lat_deg' in fields:
                item['lat_deg'] = np.float64(row['lat_deg']);

            if 'lon_deg' in fields:
                item['lon_deg'] = np.float64(row['lon_deg']);

            if 'origin_lat_deg' in fields:
                item['origin_lat_deg'] = np.float64(row['origin_lat_deg']);

            if 'origin_lon_deg' in fields:
                item['origin_lon_deg'] = np.float64(row['origin_lon_deg']);

            if 'pos_x' in fields:
                item['pos_x'] = np.float64(row['pos_x']);

            if 'pos_y' in fields:
                item['pos_y'] = np.float64(row['pos_y']);

            data_list.append(item);

    return data_list

def main():


    # Print instructions
    print("############################################################")
    print("Camera Manual Calibration")
    print("############################################################\n")

    # ##########################################################
    # # Parse Arguments
    # ##########################################################
    argparser = argparse.ArgumentParser(
        description='Camera Calibration Manual Mode')
    argparser.add_argument(
        '-i', '--init',
        action='store_true',
        help='Generate config file')
    argparser.add_argument(
        '-cfg', '--config',
        default='CAMERA_CALIB_MANUAL_CFG.ini',
        help='Path to the config file')
    args = argparser.parse_args();

    if args.init:

        # Create csv templates
        write_default_cartesian_csv('camera_calib_manual_cartesian.csv');
        write_default_latlon_csv('camera_calib_manual_latlon.csv');

        # Create default config file
        create_default_cfg();

        print('Please fill the config files and restart the program:\n-CAMERA_CALIB_MANUAL_CFG.ini\n-camera_calib_manual_cartesian.csv or camera_calib_manual_latlon.csv')
        return;

    # ##########################################################
    # # Read config file:
    # ##########################################################
    config = cfgutil.read_cfg(args.config);

    # Create output folder if needed
    ROOT_DIR = os.getcwd();
    OUTPUT_CFG_FOLDER = os.path.join(ROOT_DIR,'calib_file');
    os.makedirs(OUTPUT_CFG_FOLDER, exist_ok=True)


    ###################################################################
    # INPUT VALUES
    ###################################################################

    # Read csv:
    try:
        data_csv = read_csv(config['INPUT_PATH']['CALIB_POINT_PATH']);
    except Exception as e:
        print('[Error]: CALIB_POINT_PATH {}'.format(e))
        return;

    # If no csv or not enough data:
    if not data_csv or len(data_csv) < 4:
        print('[Error]: Not enough data in the input csv (minimum 4 points) {}'.format(config['INPUT_PATH']['CALIB_POINT_PATH']))
        return;

    # Detect if in latlon or Cartesian mode:
    csv_latlon = False;
    if 'lat_deg' in data_csv[0]:
        print('Mode: csv in latlon mode')
        csv_latlon = True;

    # Create the pixel points vector from csv
    image_points = np.zeros([1,2], dtype="double");
    first_init = True;
    for data in data_csv:

        if first_init:
            image_points[0,:] = (data['pixel_x'], data['pixel_y']);
            first_init = False;

        else:

            d = np.array([[data['pixel_x'], data['pixel_y']]], dtype= 'double');
            image_points = np.append( image_points, d , axis=0);


    # In lat/lon mode, input are given in Latitude/Longitude (degrees)
    if csv_latlon:

        # Create the latlon points vector from csv
        latlon_points = np.zeros([1,2], dtype="double");
        first_init = True;

        # Go through line of the csv
        for data in data_csv:

            # For the first line, fill directly the first row of latlon_points
            if first_init:
                latlon_points[0,:] = (data['lat_deg'], data['lon_deg']);
                latlon_origin = np.array([data['origin_lat_deg'], data['origin_lon_deg']],  dtype= 'double');
                first_init = False;

            # For the next row, append new row to latlon_points
            else:
                d = np.array([[data['lat_deg'], data['lon_deg']]], dtype= 'double');
                latlon_points = np.append( latlon_points, d , axis=0);

        # Convert the lat-lon array in F_NED array of points
        model_points_F = calib_utils.convert_latlon_F(latlon_origin, latlon_points);

    else:

        # Create the model_points_F points vector from csv
        model_points_F = np.zeros([1,3], dtype="double");
        first_init = True;
        for data in data_csv:

            if first_init:
                model_points_F[0,:] = (data['pos_x'], data['pos_y'], 0.0);
                first_init = False;

            else:

                d = np.array([[data['pos_x'], data['pos_y'], 0.0]], dtype= 'double');
                model_points_F = np.append( model_points_F, d , axis=0);


    # Print instructions
    print("Camera Calibration Software \n")

    print("Calibration details:\n")

    # Open image
    image_path = config['INPUT_PATH']['CAMERA_IMG_PATH'];
    im = cv2.imread(image_path);
    if im is None:
        raise ValueError('CAMERA_IMG_PATH is not valid: {}'.format(config['INPUT_PATH']['CAMERA_IMG_PATH']));

    # Satellite Mode:
    sat_mode = bool(config['OPTIONS']['SATELLITE_MODE'] != '0');

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    im_size = im.shape; # Getting Image size
    rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs, image_points_reproj = calib_utils.find_camera_params_opt(im_size, image_points, model_points_F, sat_mode);

    # Convert rotation vector in rotation matrix
    rot_CF_F = cv2.Rodrigues(rot_vec_CF_F)[0];
    print('rot_CF_F: ')
    print(rot_CF_F)

    # Convert rotation matrix in euler angle:
    euler_CF_F = mathutil.rotationMatrixToEulerAngles(rot_CF_F);
    print('euler_CF_F: ')
    print(euler_CF_F)

    # Position of the origin expresssed in CF
    print('trans_CF_F (position of the origin expressed in CF): ')
    print(trans_CF_F);

    im = calib_utils.display_NED_frame(im, rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs);

    im = calib_utils.display_keypoints(im, image_points_reproj, image_points);

    # Save image with key-points
    im_calib_path = image_path.split('.')[0] + '_calib.' + image_path.split('.')[-1];
    cv2.imshow("Output", im)

    print("\nInstruction:")
    print("    - Press Enter to save the calibration file")
    print("    - Press any other key to exit without saving calibration file \n")

    key = cv2.waitKey(0);
    save_bool = False;
    # if the 'Enter' key is pressed, end of the program
    if key == 13:
        save_bool = True;

    if save_bool:
        # Define output name
        image_path_name = image_path.split('/')[-1];
        output_path = image_path_name.split('.')[0] + '_cfg.yml';
        output_path = os.path.join(OUTPUT_CFG_FOLDER, output_path);

        # Save the parameters
        calib_utils.save_camera_calibration(output_path, rot_CF_F, trans_CF_F, camera_matrix, dist_coeffs);

        cv2.imwrite(im_calib_path, im );
        print('\n Image config file saved %s \n' %(im_calib_path))

    cv2.destroyAllWindows()
    print("Program Exit\n")
    print("############################################################\n")

if __name__ == '__main__':


    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    # except Exception as e:
    #     print('[Error]: {}'.format(e))
