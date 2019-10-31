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
import shutil

from traj_ext.tracker.cameramodel import CameraModel

from traj_ext.camera_calib import calib_utils
from traj_ext.utils import cfgutil
from traj_ext.utils import mathutil

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

def run_calib_manual(calib_points_path, image_path, sat_mode, output_folder, auto_save = False):

    ###################################################################
    # INPUT VALUES
    ###################################################################

    # Read csv:
    try:
        data_csv = read_csv(calib_points_path);
    except Exception as e:
        print('[Error]: reading calib points {}'.format(e))
        return;

    # If no csv or not enough data:
    if not data_csv or len(data_csv) < 4:
        print('[Error]: Not enough data in the input csv (minimum 4 points) {}'.format(calib_points_path))
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

    # Satellite Mode:
    print("Satellite Mode: {}\n".format(sat_mode))

    print("Calibration details:\n")

    # Open image
    im = cv2.imread(image_path);
    if im is None:
        raise ValueError('[ERROR]: Image path is not valid: {}'.format());

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

    cam_model = CameraModel(rot_CF_F, trans_CF_F, camera_matrix, dist_coeffs);

    im = cam_model.display_NED_frame(im);

    im = calib_utils.display_keypoints(im, image_points_reproj, image_points);

    # Save image with key-points
    cv2.imshow("Output", im)

    print("\nInstruction:")
    print("    - Press Enter to save the calibration file")
    print("    - Press any other key to exit without saving calibration file \n")

    save_bool = False;

    if auto_save:
        save_bool = True;

    else:
        key = cv2.waitKey(0);

        # if the 'Enter' key is pressed, end of the program
        if key == 13:
            save_bool = True;


    if not os.path.isdir(output_folder):
        output_folder = os.path.dirname(image_path);

    if save_bool:

        # Save the calibration points with the camera model
        calib_points_path_name = calib_points_path.split('/')[-1];
        try:
            shutil.copyfile(calib_points_path, os.path.join(output_folder, calib_points_path_name))
        except Exception as e:
            print('[Error]: Copying point file: {}'.format(e))

        # Save the camera model
        cam_file_name = image_path.split('/')[-1];
        cam_file_name = cam_file_name.split('.')[0] + '_cfg.yml';
        output_path = os.path.join(output_folder, cam_file_name);

        cam_model.save_to_yml(output_path);

        # Save the camera model
        im_calib_file_name = image_path.split('/')[-1];
        im_calib_file_name = cam_file_name.split('.')[0] + '_calib.png';

        im_calib_path = os.path.join(output_folder, im_calib_file_name);
        cv2.imwrite(im_calib_path, im );

        print('\n Image config file saved %s \n' %(im_calib_path))

    cv2.destroyAllWindows()
    print("Program Exit\n")
    print("############################################################\n")


def main():


    # Print instructions
    print("############################################################")
    print("Camera Manual Calibration")
    print("############################################################\n")

    # ##########################################################
    # # Parse Arguments
    # ##########################################################
    argparser = argparse.ArgumentParser(
        description='Camera Calibration Manual Mode:\nCalibrate the camera using keypoints matching pairs (pixel/LatLon or pixel/NEDposition)')
    argparser.add_argument(
        '-init', dest="init",
        action='store_true',
        help='Generate inputs csv files to fill with matching keypoints pairs')
    argparser.add_argument(
        '-calib_points', dest="calib_points_path",
        default='camera_calib_manual_cartesian.csv',
        help='Path to the calibration points csv file: camera_calib_manual_cartesian.csv or camera_calib_manual_latlon.csv')
    argparser.add_argument(
        '-image', dest="image_path",
        default='',
        help='Path to the image file')
    argparser.add_argument(
        '-satellite', dest="satellite_mode",
        action ='store_true',
        help='satellite mode (fixed focal length to avoid ill-posed optimization) to calibrate satellite images')
    argparser.add_argument(
        '-output_folder', dest="output_folder_path",
        default='',
        help='Output folder')

    args = argparser.parse_args();

    if args.init:

        # Create csv templates
        write_default_cartesian_csv('camera_calib_manual_cartesian.csv');
        write_default_latlon_csv('camera_calib_manual_latlon.csv');

        # # Create default config file
        # create_default_cfg();

        print('Please fill the config files and restart the program:\n-camera_calib_manual_cartesian.csv\n-camera_calib_manual_latlon.csv')
        return;

    #Run camera calibration
    run_calib_manual(args.calib_points_path, args.image_path, args.satellite_mode, args.output_folder_path);

if __name__ == '__main__':


    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    # except Exception as e:
    #     print('[Error]: {}'.format(e))
