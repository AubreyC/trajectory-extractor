# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-21 18:23:24
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-27 22:39:54

import os

from traj_ext.utils import cfgutil

from traj_ext.postprocess_track import trajutil
from traj_ext.postprocess_track import trajectory

# Define output dir for the test
DIR_PATH = os.path.dirname(__file__);
OUTPUT_DIR_PATH = os.path.join(DIR_PATH,'test_output');

def test_time_list_csv():

    # Generate time_ms list
    list_times_ms = [];
    for i in range(100):
        list_times_ms.append(i*15);

    # Create outut dir if not created
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

    # Write list time ms
    path_to_csv = trajutil.write_time_list_csv(OUTPUT_DIR_PATH, 'test', list_times_ms);

    # Write list time ms
    list_times_ms_copy = trajutil.read_time_list_csv(path_to_csv);

    # Check if true
    assert (list_times_ms == list_times_ms_copy);

def test_trajectory_csv():

    # Read config file:
    traj_csv_path = 'test_dataset/brest_20190609_130424_327_334/output/vehicles/traj/csv/brest_20190609_130424_327_334_traj.csv'

    # Run the det association:
    traj_list = trajectory.Trajectory.read_trajectory_panda_csv(traj_csv_path);

# def test_display_traj_timestamp():

#     display_traj_timestamp(traj, time_ms, img, cam_model, color_text = (0,0,255));


if __name__ == '__main__':

    try:
        test_time_list_csv();
        test_trajectory_csv();
        # test_display_traj_timestamp();

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

