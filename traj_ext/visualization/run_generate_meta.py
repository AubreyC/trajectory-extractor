# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-24 15:34:17
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-16 21:28:47

import copy
import cv2
import argparse
import os
import subprocess
import math;

from traj_ext.postprocess_track import trajutil
from traj_ext.postprocess_track.trajectory import Trajectory
from traj_ext.postprocess_track.trajectory import Trajectory

from traj_ext.utils import mathutil

def main():

    # Arg parser:
    parser = argparse.ArgumentParser(description='Generate meta data for a trajectories csv');
    parser.add_argument('-traj', dest="traj_csv_path", type=str, help='Path of the trajectories csv', default='');
    parser.add_argument('-time', dest="list_times_ms_path", type=str, help='Path of the time csv', default='');
    parser.add_argument('-output', dest="output_folder", type=str, help='Folder to save output images', default ='');
    parser.add_argument('-location_name', dest="location_name", type=str, help='Location Name', default ='');
    parser.add_argument('-date', dest="date", type=str, help='Date: YYYYMMDD', default ='');
    parser.add_argument('-start_time', dest="start_time", type=str, help='Start time: HHMMSS', default ='');
    parser.add_argument('-delta_ms', dest="delta_ms", type=int, help='Delta time between points in ms', default = 100);

    args = parser.parse_args()

    if args.output_folder == '':
        print('Error: Output path if empty {}'.format(args.output_folder))
        return;

    # Read traj
    traj_list = trajutil.read_trajectory_panda_csv(args.traj_csv_path);
    list_times_ms = trajutil.read_time_list_csv(args.list_times_ms_path);


    # Write meta data
    duration_s = 0;
    if len(list_times_ms) > 2:
        duration_s = float(list_times_ms[-1] - list_times_ms[0])/float(1e3);

    total_traj_nb, total_traj_time_s, total_traj_distance_m = Trajectory.generate_metadata(traj_list);
    print('Total traj: {} Total time: {}s Total distance: {}m'.format(total_traj_nb, total_traj_time_s, total_traj_distance_m));

    # Vehicle trajectory
    name_prefix = args.traj_csv_path.split('/')[-1];
    name_prefix = name_prefix.split('.')[0];

    trajutil.write_trajectory_meta(args.output_folder, name_prefix, args.location_name,\
                                                          args.date,\
                                                          args.start_time,\
                                                          duration_s,
                                                          str(args.delta_ms),
                                                          total_traj_nb,\
                                                          total_traj_time_s,\
                                                          total_traj_distance_m);


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
