#################################################################################
#
# Time Interval object used to ignore some time period
#
#################################################################################

import numpy as np
import os
import sys
import cv2
import pandas as pd
import collections


class TimeIgnore(object):

    """Time Interval object used to ignore some time period"""
    def __init__(self, time_ms_start, time_ms_end):

        if time_ms_start >= time_ms_end:
            raise ValueError('Time Ignore: time_ms_start:{} >= time_ms_end: {}'.format(time_ms_start, time_ms_end));

        self.time_ms_start = int(time_ms_start);
        self.time_ms_end = int(time_ms_end);


    def check_time_inside(self, time_ms):
        """Check if time is inside the TimeIgnore interval

        Args:
            time_ms (TYPE): Time to check in ms

        Returns:
            TYPE: Bool
        """
        result = False;
        if time_ms >= self.time_ms_start and time_ms <= self.time_ms_end:
            result = True;

        return result;

    @classmethod
    def to_csv(cls, path_to_csv, time_ignore_list):
        """Write list of time interval to ingore

        Args:
            path_to_csv (TYPE): Path to the csv
            time_ignore_list (TYPE): List of time interval to ignore

        Returns:
            TYPE: Path to the csv
        """
        # Create empty dict
        dict_time_ignore= collections.OrderedDict.fromkeys(['time_ms_start', 'time_ms_end']);
        dict_time_ignore['time_ms_start'] = [];
        dict_time_ignore['time_ms_end'] = [];


        # Put data
        for time_ignore in time_ignore_list:
            dict_time_ignore['time_ms_start'].append(time_ignore.time_ms_start);
            dict_time_ignore['time_ms_end'].append(time_ignore.time_ms_end);

        # Create dataframe
        df_time_ignore = pd.DataFrame(dict_time_ignore);

        # Sort by track_id:
        df_time_ignore.sort_values(by=['time_ms_start'], inplace = True);

        # Write dataframe in csv
        df_time_ignore.to_csv(path_to_csv, index=False);

        return path_to_csv;

    @classmethod
    def from_csv(cls, path_to_csv):
        """Read time interval list from csv file

        Args:
            path_to_csv (TYPE): Path to the csv

        Returns:
            TYPE: List of time interval
        """

        # Read dataframe with panda
        df = pd.read_csv(path_to_csv);

        time_ignore_list = [];
        for index, row in df.iterrows():

            # print(row['track_id']);
            # print(row['agent_type']);

            time_ms_start = int(row['time_ms_start']);
            time_ms_end = int(row['time_ms_end']);

            time_ignore = TimeIgnore(time_ms_start, time_ms_end);
            time_ignore_list.append(time_ignore);

        return time_ignore_list;

    @classmethod
    def check_time_inside_list(cls, time_ignore_list, time_ms):
        """Check if time is inside any of the time interval from a list of time intervals

        Args:
            time_ignore_list (TYPE): List of time intervals
            time_ms (TYPE): Time to check in ms

        Returns:
            TYPE: Boolean: True if in one or more time interval
        """

        time_inside = False;
        for t_ignore in time_ignore_list:

            time_inside = t_ignore.check_time_inside(time_ms);
            if time_inside:
                break;

        return time_inside;
