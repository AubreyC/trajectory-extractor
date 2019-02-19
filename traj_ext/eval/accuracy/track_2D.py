# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-11-26 17:45:38
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

########################################################################################
#
# Track for Tracking accuracy evaluation. Holds detetions in 2D
#
########################################################################################


import numpy as np;
import matplotlib.pyplot as plt
from  math import *;
import os
import cv2
import copy
import sys
import time
import operator
import abc


from tracker import cameramodel as cm
from utils.mathutil import *
from tracker.utils import tracker_utils

from postprocess_track import trajutil
from postprocess_track import trajectory

class MeasBox2D():
    def __init__(self, time_ms, label, xtl, ytl, xbr, ybr):
        self.time_ms = time_ms;
        self.label = label;
        self.xtl = xtl;
        self.ytl = ytl;
        self.xbr = xbr;
        self.ybr = ybr;
        self.roi = [ytl, xtl, ybr, xbr];
        self.center = np.array([float(xtl + xbr)/2, float(ytl + ybr)/2])
        self.center.shape = (2,1);

class Track_2D(object):

    def __init__(self, track_id, label=None):

        # Tracker ID:
        self._track_id = int(track_id);
        self._label = label;

        # Store past measurement
        self._box2D_meas_list = [];

    def push_2Dbox_meas(self, time_ms, label, xtl, ytl, xbr, ybr):

        # Add to 3D box meas list:
        box2D_meas_data = MeasBox2D(time_ms, label, xtl, ytl, xbr, ybr);
        self._box2D_meas_list.append(box2D_meas_data);

        return;

    def get_id(self):
        return self._track_id;

    def get_label(self):
        return self._label;

    def set_label(self, label):
        self._label = label;

    def _create_list_time(self, full_times_ms, init_time_ms, end_time_ms):

        try:
            i_start = full_times_ms.index(init_time_ms);
            i_end = full_times_ms.index(end_time_ms);
        except Exception as e:
            print("[ERROR]: Track_postprocess _create_list_time init_time_ms or end_time_ms not found");
            raise

        l_times_ms = full_times_ms[i_start:i_end + 1];

        return l_times_ms;

    def _get_index_box2D_meas(self, time_ms):

        index = None;
        for i, meas in enumerate(self._box2D_meas_list):
            if meas.time_ms == time_ms:
                index = i;
                break;

        return index;

    def get_box2D_at_timestamp(self, time_ms):

        result = None;

        # Making sure traj is not empty
        if self.get_length_ms() > 0:

            # First check if timestamp is withing the time frame of the trajetcory
            if time_ms > self._box2D_meas_list[0].time_ms and time_ms < self._box2D_meas_list[-1].time_ms:

                # Look for specific point
                for data in self._box2D_meas_list:
                    if data.time_ms == time_ms:
                        result = data;
                        break;

        return result;


    def compute_label(self):

        label_dict = {};

        index = None;
        for i, meas in enumerate(self._box2D_meas_list):
            label = meas.label;

            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1

        label_max = max(label_dict.items(), key=operator.itemgetter(1))[0]

        return label_max;


    def get_length_ms(self):

        l = 0;
        if len(self._box2D_meas_list) > 1:

            l = self._box2D_meas_list[-1].time_ms - self._box2D_meas_list[0].time_ms;

        if l < 0:
            print('[ERROR Track 2D]: get_length_ms < 0: {}'.format(l));

        return l;


    # def add_jumped_meas(self, list_time_ms):

    #     if len(self._box2D_meas_list) < 2:
    #         return;

    #     init_time_ms = self._box2D_meas_list[0].time_ms;
    #     end_time_ms = self._box2D_meas_list[0].time_ms;

