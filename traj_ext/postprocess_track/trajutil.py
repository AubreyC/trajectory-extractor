# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-04-05 09:50:35
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

####################################################################
#
# Utils functions for trajectory logging
#
####################################################################

import copy
import numpy as np
import csv
import os
import math
import cv2
import sys


from postprocess_track import trajectory
from tracker import cameramodel as cm

def plot_traj_on_images(traj_list, time_ms, img_street, img_sat, cam_model_street, cam_model_sat, color_text = (0,0,255)):

    for traj in traj_list:

        # =========================================
        #               Plot filtering
        # ==========================================

        traj_point = traj.get_point_at_timestamp(time_ms);

        if not (traj_point is None):

            if not (img_sat is None):

                # print('time_ms: {} Traj id: {} x: {} y: {}'.format(time_ms, traj.get_id(), traj_point.x, traj_point.y));

                # Reproject on the satellite image
                pt_pix = cam_model_sat.project_points(np.array([(traj_point.x, traj_point.y, 0.0)]));
                pt_pix = (int(pt_pix[0]),int(pt_pix[1]));
                img_sat = cv2.circle(img_sat, pt_pix,3, traj.get_color(), -1);

                # Add annotations to the track:
                text = "id: %i" % (traj.get_id());
                img_sat = cv2.putText(img_sat, text, pt_pix, cv2.FONT_HERSHEY_COMPLEX, 0.8, color_text, 1)

            if not (img_street is None):

                # Reproject on the satellite image
                pt_pix = cam_model_street.project_points(np.array([(traj_point.x, traj_point.y, 0.0)]));
                pt_pix = (int(pt_pix[0]),int(pt_pix[1]));
                cv2.circle(img_street, pt_pix,3, traj.get_color(), -1);

                # Add annotations to the track:
                text = "id: %i" % (traj.get_id());
                img_street = cv2.putText(img_street, text, pt_pix, cv2.FONT_HERSHEY_COMPLEX, 0.8, color_text, 1)

    return img_sat, img_street

# Write the box detected in csv, one csv per frame
def write_traj_csv(path_csv, agent_results):

    with open(path_csv, 'w') as csvfile:

        fieldnames = [];

        # Add new keys
        fieldnames.append('id');
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

        for index in range(0,len(agent_results)):

            # Field management
            dict_row = {};

            # Data:
            dict_row['id'] = agent_results[index]['id'];
            dict_row['timestamp_ms'] = agent_results[index]['timestamp_ms'];
            dict_row['agent_type'] = agent_results[index]['agent_type'];
            dict_row['x'] = agent_results[index]['x'];
            dict_row['y'] = agent_results[index]['y'];
            dict_row['vx'] = agent_results[index]['vx'];
            dict_row['vy'] = agent_results[index]['vy'];
            dict_row['psi_rad'] = agent_results[index]['psi_rad'];

            writer.writerow(dict_row);

def read_traj_list_from_csv(traj_path):

    # Open img folder
    list_traj_file = os.listdir(traj_path);
    list_traj_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))));

    # List of trajectories:
    traj_list = [];

    for current_index, current_traj_file in enumerate(list_traj_file):

        data_list_csv = read_traj_csv(os.path.join(traj_path, current_traj_file));

        for data_csv in data_list_csv:

            traj = get_traj_in_list(traj_list, data_csv['id']);

            # If trajectory is not created yet, create it
            if traj is None:
                traj = trajectory.Trajectory(data_csv['id'], data_csv['agent_type']);
                traj_list.append(traj);

            # Add point to the corresponding trajetcory
            traj.add_point(data_csv['timestamp_ms'], \
                           data_csv['x'], \
                           data_csv['y'], \
                           data_csv['vx'], \
                           data_csv['vy'], \
                           data_csv['psi_rad']);

    return traj_list;

def get_traj_in_list(traj_list, id):

    for traj in traj_list:
        if id == traj.get_id():
            return traj;

    return None;

def write_trajectory_csv(folder_path, name_prefix, traj_list, list_times_ms):

    for index, timestamp_ms in enumerate(list_times_ms):

        index_str =  str(index).zfill(8);
        csv_name = name_prefix +'_' + index_str + '.csv';
        path_csv = os.path.join(folder_path, csv_name);

        with open(path_csv, 'w') as csvfile:

            fieldnames = [];

            # Add new keys
            fieldnames.append('id');
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
                    dict_row['id'] = traj.get_id();
                    dict_row['timestamp_ms'] = traj_point.time_ms;
                    dict_row['agent_type'] = traj.get_agent_type();
                    dict_row['x'] = traj_point.x;
                    dict_row['y'] = traj_point.y;
                    dict_row['vx'] = traj_point.vx;
                    dict_row['vy'] = traj_point.vy;
                    dict_row['psi_rad'] = traj_point.psi_rad;

                    writer.writerow(dict_row);


def read_traj_csv(csv_path):

    data_list = [];

    # Create dict
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:

            fields = row.keys();
            item = {};

            if 'id' in fields:
                item['id'] = int(row['id']);

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


