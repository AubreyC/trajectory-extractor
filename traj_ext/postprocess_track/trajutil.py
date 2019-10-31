# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-04-05 09:50:35
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

####################################################################
#
# Utils functions for trajectory
#
####################################################################

import copy
import numpy as np
import csv
import os
import math
import cv2
import sys
import pandas as pd
import collections

from traj_ext.postprocess_track import trajectory
from traj_ext.tracker import cameramodel as cm
from traj_ext.box3D_fitting import Box3D_utils

def display_traj_list_on_image(traj_list, time_ms, image, cam_model):
    """Display a lits of trajectory on images

    Args:
        traj_list (TYPE): List of Trajectory object
        time_ms (TYPE): Time to display in ms
        image (TYPE): Image
        cam_model (TYPE): Camera Model

    Returns:
        TYPE: Image with trajectory
    """
    for traj in traj_list:

            image, _ = traj.display_on_image(time_ms, image, cam_model);

    return image;

def read_traj_seperate_csv(folder_path):
    """Read trajectories from seperate csv files: One csv by frame

    Args:
        folder_path (TYPE): Path to the folder containing the csv

    Returns:
        TYPE: List of trajectories
    """

    # Open img folder
    list_traj_file = os.listdir(folder_path);
    list_traj_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))));

    # List of trajectories:
    traj_list = [];

    for current_index, current_traj_file in enumerate(list_traj_file):

        data_list_csv = read_traj_csv(os.path.join(folder_path, current_traj_file));

        for data_csv in data_list_csv:

            traj = get_traj_in_list(traj_list, data_csv['track_id']);

            # If trajectory is not created yet, create it
            if traj is None:
                traj = trajectory.Trajectory(data_csv['track_id'], data_csv['agent_type']);
                traj_list.append(traj);

            # Add point to the corresponding trajetcory
            traj.add_point(data_csv['timestamp_ms'], \
                           data_csv['x'], \
                           data_csv['y'], \
                           data_csv['vx'], \
                           data_csv['vy'], \
                           data_csv['psi_rad']);

    return traj_list;


def read_traj_csv(csv_path):
    """Read trajectory single csv from separte csv trajectories.

    Args:
        csv_path (TYPE): Description

    Returns:
        TYPE: List of dictionnary
    """

    data_list = [];

    # Create dict
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:

            fields = row.keys();
            item = {};

            if 'track_id' in fields:
                item['track_id'] = int(row['track_id']);

            if 'timestamp_ms' in fields:
                item['timestamp_ms'] = int(float(row['timestamp_ms']));

            if 'agent_type' in fields:
                item['agent_type'] = str(row['agent_type']);

            if 'x' in fields:
                item['x'] = np.float32(row['x']);

            if 'y' in fields:
                item['y'] = np.float32(row['y']);

            if 'vx' in fields:
                item['vx'] = np.float32(row['vx']);

            if 'vy' in fields:
                item['vy'] = np.float32(row['vy']);

            if 'psi_rad' in fields:
                item['psi_rad'] = np.float32(row['psi_rad']);

            data_list.append(item);

    return data_list

def write_traj_seperate_csv(folder_path, name_prefix, traj_list, list_times_ms):
    """Write list of trajectories in seperate csv file, one csv by frame.

    Args:
        folder_path (TYPE): Description
        name_prefix (TYPE): Description
        traj_list (TYPE): Description
        list_times_ms (TYPE): Description

    Raises:
        NameError: Description
    """
    for index, timestamp_ms in enumerate(list_times_ms):

        index_str =  str(index).zfill(8);
        csv_name = name_prefix +'_' + index_str + '.csv';
        path_csv = os.path.join(folder_path, csv_name);

        with open(path_csv, 'w') as csvfile:

            fieldnames = [];

            # Add new keys
            fieldnames.append('track_id');
            fieldnames.append('timestamp_ms');
            fieldnames.append('agent_type');
            fieldnames.append('x');
            fieldnames.append('y');
            fieldnames.append('vx');
            fieldnames.append('vy');
            fieldnames.append('psi_rad');

            #Write field name
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
            writer.writeheader();

            for traj in traj_list:

                # Define name:
                traj_point = traj.get_point_at_timestamp(timestamp_ms);
                if not(traj_point is None):

                    # Field management
                    dict_row = {};

                    if not (timestamp_ms == traj_point.time_ms):
                        raise NameError('[ERROR]: write_trajectory_csv (timestamp_ms != traj.time_ms)')

                    # Data:
                    dict_row['track_id'] = traj.get_id();
                    dict_row['timestamp_ms'] = traj_point.time_ms;
                    dict_row['agent_type'] = traj.get_agent_type();
                    dict_row['x'] = traj_point.x;
                    dict_row['y'] = traj_point.y;
                    dict_row['vx'] = traj_point.vx;
                    dict_row['vy'] = traj_point.vy;
                    dict_row['psi_rad'] = traj_point.psi_rad;

                    writer.writerow(dict_row);

def get_traj_in_list(traj_list, id):
    """Get trajectory by id

    Args:
        traj_list (TYPE): Description
        id (TYPE): Description

    Returns:
        TYPE: Description
    """
    for traj in traj_list:
        if id == traj.get_id():
            return traj;

    return None;

def get_time_ms_max(traj_list):
    """Get highest time_ms in a list of trajectories

    Args:
        traj_list (TYPE): Description

    Returns:
        TYPE: Time max in ms
    """
    time_ms_max = 0;
    for traj in traj_list:
        end_point = traj.get_end_trajoint();
        if end_point:
            time_ms_end = end_point.time_ms;

            if time_ms_end > time_ms_max:
                time_ms_max = time_ms_end;

    return time_ms_max;

def write_time_list_csv(folder_path, name_prefix, list_times_ms):
    """Write list of time msfor a list of trajectories

    Args:
        folder_path (TYPE): Description
        name_prefix (TYPE): Description
        list_times_ms (TYPE): Description

    Returns:
        TYPE: Path to the csv file
    """
    # Write trajectories
    csv_name = name_prefix +'_time_traj.csv';
    path_to_csv = os.path.join(folder_path, csv_name);

    with open(path_to_csv, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for time_ms in list_times_ms:
            writer.writerow([time_ms])

    csvFile.close()

    return path_to_csv;

def read_time_list_csv(csv_path):
    """Read list of time ms from a csv

    Args:
        csv_path (TYPE): Description

    Returns:
        TYPE: list of time ms
    """
    list_times_ms = [];
    with open(csv_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            row_int = list(map(int, row))
            list_times_ms = list_times_ms + row_int;

        csvFile.close();

    return list_times_ms

def read_traj_ignore_list_csv(csv_path):
    """Read trajectory id ignore list from a csv

    Args:
        csv_path (TYPE): Path to the csv

    Returns:
        TYPE: List of id of trajetcories to ignore
    """
    list_traj_ignore = [];
    with open(csv_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            row_int = list(map(int, row))
            list_traj_ignore = list_traj_ignore + row_int;

        csvFile.close();

    return list_traj_ignore;

def write_traj_ignore_list_csv(path_to_csv, list_traj_ignore):
    """Write trajectory id list to ingnore to a csv
    Args:
        path_to_csv (TYPE): Description
        list_traj_ignore (TYPE): Description

    Returns:
        TYPE: Description
    """

    # Write trajectories
    with open(path_to_csv, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for traj_id in list_traj_ignore:
            writer.writerow([traj_id])

    csvFile.close()

    return path_to_csv;


def write_list_csv(path_to_csv, data_list):
    """Generic function to write a list of integer in a csv

    Args:
        path_to_csv (TYPE): Description
        data_list (TYPE): List of integer

    Returns:
        TYPE: Path to the csv
    """

    # Write trajectories
    with open(path_to_csv, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for data in data_list:
            writer.writerow([data])

    csvFile.close()

    return path_to_csv;

def read_list_csv(csv_path):
    """Generic function to read a list of integer from a csv

    Args:
        csv_path (TYPE): Description

    Returns:
        TYPE: List of integer
    """

    list_data = [];
    with open(csv_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            row_int = list(map(int, row))
            list_data = list_data + row_int;

        csvFile.close();

    return list_data;


def write_trajectory_meta(folder_path, name_prefix, location_name, date, start_time, duration_s, delta_ms, total_traj_nb, total_traj_time_s, total_traj_distance_m):
    """Write trajectory meta information in a csv

    Args:
        folder_path (TYPE): Path to the folder
        name_prefix (TYPE): Name prefix
        location_name (TYPE): Location name
        date (TYPE): Date with YYYMMDD format
        start_time (TYPE): Start time in HHMMSS with 24 hour format
        duration_s (TYPE): Duration in seconds
        delta_ms (TYPE): Delta time between points in ms
        total_traj_nb (TYPE): Total number of trajectories
        total_traj_time_s (TYPE): Cumulative total time of trajectories in seconds
        total_traj_distance_m (TYPE): Cumulative total distance of trajectories in meters
    """

    # Write trajectories
    csv_name = name_prefix +'_meta.csv';
    df_meta_path = os.path.join(folder_path, csv_name);

    dict_meta = collections.OrderedDict.fromkeys(['location_name',\
                                                     'date',\
                                                     'start_time',\
                                                     'duration_s',\
                                                     'delta_ms',\
                                                     'total_traj_nb',\
                                                     'total_traj_distance_m',\
                                                     'total_traj_time_s']);

    dict_meta['location_name'] = [location_name];
    dict_meta['date'] = [date];
    dict_meta['start_time'] = [start_time];
    dict_meta['duration_s'] = [duration_s];
    dict_meta['delta_ms'] = [delta_ms];
    dict_meta['total_traj_nb'] = [total_traj_nb];
    dict_meta['total_traj_distance_m'] = [int(total_traj_distance_m)];
    dict_meta['total_traj_time_s'] = [int(total_traj_time_s)];

    # Create dataframe
    df_meta = pd.DataFrame(dict_meta);

    # Write dataframe in csv
    df_meta.to_csv(df_meta_path, index = False);

    print('Saving meta data: {}'.format(df_meta_path));

def get_name_prefix(image_name):
    """Strip image name to extract name_prefix

    Args:
        image_name (TYPE): String image name: varna_20190125_153327_240_480_0000000033.png

    Returns:
        TYPE: name prefix: varna_20190125_153327_240_480
    """
    name_prefix = None;
    if image_name:
        name_prefix = image_name.split('.')[0];
        name_prefix = name_prefix.split('_')[:-1];
        name_prefix = '_'.join(name_prefix);

    return name_prefix;

def find_closest_traj(traj, traj_list, traj_included = False):
    """Find the closest trajectory in a list of trajectories.
       Based on x and y distance at each timestep.

    Args:
        traj (TYPE): Trajectory of reference
        traj_list (TYPE): List of trajectories to serach in
        traj_included (BOOLEAN): traj is completely included in one of the trajectories in traj_list

    Returns:
        TYPE: Index of the closest trajectory in the list and the trajectory itself
    """

    results = [];
    results_index = [];
    for index, t in enumerate(traj_list):
        error = traj.compute_distance_to_traj(t, traj_included);

        if not (error is None):
            # results.append(np.linalg.norm(error));
            results.append(np.mean(error));
            if(math.isnan(np.mean(error))):
                print('Result nan: {}'.format(error))
            results_index.append(index);


    # Making sure the error value has been updated
    if not results:
        print('[ERROR]: find_closest_traj')
        return None, None;

    # print('find_closest_traj results: {}'.format(results));
    results = np.array(results);
    index_min = np.argmin(results);


    index_min_traj_list = results_index[index_min];
    return index_min_traj_list, traj_list[index_min_traj_list];

def compute_time_traj_overlap(traj_1, traj_2):

    # Get the start en end time to compare the two trajectories
    start_t = max(traj_1.get_traj()[0].time_ms, traj_2.get_traj()[0].time_ms);
    end_t = min(traj_1.get_traj()[-1].time_ms, traj_2.get_traj()[-1].time_ms);

    return start_t, end_t;


