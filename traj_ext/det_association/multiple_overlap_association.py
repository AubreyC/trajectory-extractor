# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-04-26 17:01:36
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

import os
import sys

import random
import math
import numpy as np
import colorsys

import matplotlib
import matplotlib.pyplot as plt

import cv2
from scipy.optimize import linear_sum_assignment

from traj_ext.object_det import det_object
from traj_ext.det_association.track_2D import Track2D


class MultipleOverlapAssociation(object):
    """Implement the Measurement to Track association with:
    - Overlap between detection
    - Categories
    """
    def __init__(self, associate_with_label = False, threshold_overlap = 0.3, nb_frame_past_max = 10, det_zone_IM=None):

        self.tracker_list_active = []
        self.tracker_list = []

        self.index_track = 0;
        self.frame_index = 0;

        # Flag to constraint detection association by label:
        self._no_label_association = not (bool(associate_with_label));
        self._threshold_overlap = threshold_overlap;
        self._nb_frame_past_max = nb_frame_past_max;
        self._det_zone_IM = det_zone_IM;

    def get_tracker_list(self):
        return self.tracker_list;

    def push_detection(self, det_object_list, frame):

        nb_det = len(det_object_list);
        nb_tk_overlap = len(self.tracker_list_active);
        det_instances = list(range(0, nb_det));

        track_det_list = [];

        if nb_tk_overlap > 0:

            for frame_past in range(self.frame_index,max(0, self.frame_index - self._nb_frame_past_max) -1, -1):

                # Creating cost matrix
                cost_mat = np.zeros((nb_tk_overlap, nb_det));

                for tk_ind in range(0, nb_tk_overlap):

                    for det_ind in det_instances:

                        det_object_tk = None;

                        # Get mask and ROI from tracker
                        if (not (self.tracker_list_active[tk_ind].get_last_frame_index() == self.frame_index)):

                            # Check same class_id if self._no_label_association is false:
                            if self._no_label_association or self.tracker_list_active[tk_ind].get_last_agent_type() == det_object_list[det_ind].label:
                                det_object_tk = self.tracker_list_active[tk_ind].get_det_frame_index(frame_past);

                        if not (det_object_tk is None):

                            m_tk = det_object_tk.det_mask;
                            roi_tk = det_object_tk.det_2Dbox;

                            # Get mask and ROI
                            m_det = det_object_list[det_ind].det_mask;
                            roi_det = det_object_list[det_ind].det_2Dbox;

                            # Compute cost matrix: Percentage overlap between detection in previous track and new detection
                            over = 0;
                            if det_object.intersection_rect(roi_tk, roi_det) > 0 :
                                over = det_object.intersection_over_union_mask(m_tk, m_det);

                            # print('tk_id: {} det_id: {} cost:{}'.format(self.tracker_list_active[tk_ind].track_id, det_object_list[det_ind].det_id, over))
                            cost_mat[tk_ind, det_ind] = over;

                        else:

                            # If no mask, percent overlap is 0
                            cost_mat[tk_ind, det_ind] = 0;

                # Assigning measurement to track: Find maximum overlap
                tk_ind_list, det_ind_list = linear_sum_assignment(-cost_mat);

                for tk_ind, det_ind in zip(tk_ind_list, det_ind_list):

                    # Threshold on the association:
                    if cost_mat[tk_ind, det_ind] > self._threshold_overlap:

                        # print('Assign tk_id: {} det_id: {} cost:{}'.format(self.tracker_list_active[tk_ind].track_id, det_object_list[det_ind].det_id, cost_mat[tk_ind, det_ind]))

                        # Push detection to the tracker
                        self.tracker_list_active[tk_ind].push_det(self.frame_index, det_object_list[det_ind]);
                        track_det_list.append([self.tracker_list_active[tk_ind].track_id, det_object_list[det_ind].det_id]);

                        # Remove detection from the list
                        det_instances.remove(det_ind);

        for det_i in det_instances:

            # Get info
            m_det = det_object_list[det_i].det_mask;
            roi_det = det_object_list[det_i].det_2Dbox;
            label = det_object_list[det_i].label;

            # Create new 2D track
            tk_o = Track2D(self.index_track, delete_past_mask=self._nb_frame_past_max);
            tk_o.push_det(self.frame_index, det_object_list[det_i]);

            # Add new track2D to list
            self.tracker_list_active.append(tk_o);
            self.tracker_list.append(tk_o);

            self.index_track = self.index_track + 1;

            # Actually keep the original det_indice
            track_det_list.append([tk_o.track_id, det_object_list[det_i].det_id]);

        # Set to unactive tracker
        for tk_o in list(self.tracker_list_active):

            # If no detections for the past XXX frames
            if (self.frame_index - tk_o.get_last_frame_index()) > self._nb_frame_past_max:

                tk_o.active = False;
                self.tracker_list_active.remove(tk_o);
                continue;

            # If tracker is out of detection zone
            if not (self._det_zone_IM is None):

                det = tk_o.get_det_frame_index(self.frame_index);
                if not (det is None):

                    if not self._det_zone_IM.in_zone(det.get_center_det_2Dbox()):
                        tk_o.active = False;
                        self.tracker_list_active.remove(tk_o);

        # Keep track of frame
        self.frame_index = self.frame_index + 1;

        return track_det_list;