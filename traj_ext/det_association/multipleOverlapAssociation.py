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

from object_det.mask_rcnn import detect_utils
from box3D_fitting import Box3D_utils
from det_association import overlapTrack as tk_over

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

class MultipleOverlapAssociation(object):
    """Implement the Measurement to Track association with:
    - Overlap between detection
    - Categories
    """
    def __init__(self, associate_with_label = False, threshold_overlap = 0.3, nb_frame_past_max = 10):

        self.tracker_list = []
        self.index_track = 0;
        self.frame_index = 0;

        # Flag to constraint detection association by label:
        self._no_label_association = not (bool(associate_with_label));
        self._threshold_overlap = threshold_overlap;
        self._nb_frame_past_max = nb_frame_past_max;

    def push_detection(self, det_result):

        nb_det = len(det_result['rois']);
        nb_tk_overlap = len(self.tracker_list);
        det_instances = list(range(0, nb_det));

        track_number = list(range(0,nb_det));

        colors = random_colors(nb_det);

        if nb_tk_overlap > 0:

            for frame_past in range(0,min(self._nb_frame_past_max, self.frame_index)+1):

                # Creating cost matrix
                cost_mat = np.zeros((nb_tk_overlap, nb_det));

                for tk_ind in range(0, nb_tk_overlap):

                    for det_ind in det_instances:

                        # Init to None:
                        m_tk = None;
                        roi_tk = None;

                        # Get mask and ROI from tracker
                        if (not (self.tracker_list[tk_ind].last_det_index == self.frame_index)):

                            # Check same class_id if self._no_label_association is false:
                            if self._no_label_association or self.tracker_list[tk_ind].class_id == det_result['class_ids'][det_ind]:

                                m_tk = self.tracker_list[tk_ind].get_mask(frame_past);
                                roi_tk = self.tracker_list[tk_ind].get_roi(frame_past);

                        if not (m_tk is None):

                            # Get mask and ROI
                            m_det = det_result['masks'][:,:,det_ind]
                            # m_det = det_result['mask_cont'][det_ind]
                            roi_det = det_result['rois'][det_ind]

                            # Compute cost matrix: Percentage overlap between detection in previous track and new detection
                            cost_mat[tk_ind, det_ind] = Box3D_utils.overlap_percentage_mask(m_tk, m_det);

                            # cost_mat[tk_ind, det_ind] = tk_over.compute_overlap_percentage_cont(roi_tk, roi_det, m_tk, m_det);

                        else:

                            # If no mask, percent overlap is 0
                            cost_mat[tk_ind, det_ind] = 0;

                # Assigning measurement to track: Find maximum overlap
                tk_ind_list, det_ind_list = linear_sum_assignment(-cost_mat);

                for tk_ind, det_ind in zip(tk_ind_list, det_ind_list):

                    # Threshold on the association:
                    if cost_mat[tk_ind, det_ind] > self._threshold_overlap:

                        # Add mask to the tracker
                        m_det = det_result['masks'][:,:,det_ind];
                        # m_det = det_result['mask_cont'][det_ind]
                        roi_det =  det_result['rois'][det_ind]

                        self.tracker_list[tk_ind].push_det(roi_det, m_det, self.frame_index);

                        # Colors the detection with the track color
                        colors[det_ind] = self.tracker_list[tk_ind].color
                        track_number[det_ind] = self.tracker_list[tk_ind].id;

                        # Remove detection from the list
                        det_instances.remove(det_ind);

            for tk_o in self.tracker_list:
                if not (tk_o.last_det_index == self.frame_index):
                    tk_o.push_det_none();


        for det_i in det_instances:

            m_det = det_result['masks'][:,:,det_i];
            # m_det = det_result['mask_cont'][det_i]

            roi_det = det_result['rois'][det_i]
            class_id = det_result['class_ids'][det_i];

            tk_o = tk_over.OverlapTrack(self.index_track, roi_det, m_det, class_id, self.frame_index, self._nb_frame_past_max);
            tk_o.color = colors[det_i];
            self.tracker_list.append(tk_o);
            self.index_track = self.index_track + 1;
            track_number[det_i] = tk_o.id;

        # Delete unactive tracker
        for tk_o in list(self.tracker_list):
            if (self.frame_index - tk_o.last_det_index) > 10:
                self.tracker_list.remove(tk_o);

        # Keep track of frame
        self.frame_index = self.frame_index + 1;

        return track_number, colors;