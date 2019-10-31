# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-03-21 14:12:39
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

import os
import sys

import random
import math
import numpy as np
import time

import csv
import configparser
import argparse
from shutil import copyfile
import pickle
import csv

import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from traj_ext.det_association import multiple_overlap_association
from traj_ext.object_det import det_object

from traj_ext.utils import cfgutil
from traj_ext.utils import det_zone

class TrackMerge(object):
    """Merge incomplete tracks based on a visual tracker: CSRT tracker (OpenCV)"""


    OVERLAP_MIN_CRST = 0.3;
    """ Overlap minimum between ROI CSRT tracker and ROI of tracker candidate"""

    HORIZON_CSRT = 120;
    """ Horizon used for the CSRT tracker to find merging track"""

    PREVIOUS_HORIZON_CSRT = 4;
    """ Start the CSRT tracker before the end of the track to better track the object after the track ends"""

    MATCHING_THRESH = 5;
    """  Merge tracks if CRST tracker and candidate matches for this number of frame """

    @classmethod
    def save_track_merge_csv(cls, path_to_csv, tk_match_list):
        """Write the list of tracks that should be merged in a csv

        Args:
            path_to_csv (string): Path to the csv
            tk_match_list (list): List of tracks id that should be merged
        """

        # Write the merging list intot a csv
        with open(path_to_csv, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(tk_match_list)

        csvFile.close()

    @classmethod
    def read_track_merge_csv(cls, path_to_csv):
        """Read the list of tracks that should be merged

        Args:
            path_to_csv (string): Path to the csv

        Returns:
            TYPE:  List of tracks id that should be merged
        """
        tk_merge_list = [];
        with open(path_to_csv, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                try:
                    row_int = list(map(int, row))
                except Exception as e:
                    print("[Error]: reading track merge at row {}".format(row));
                    raise(e)

                tk_merge_list.append(row_int);

            csvFile.close();

        return tk_merge_list

    @classmethod
    def convert_roi_to_bbox(cls, roi):
        x_1 = int(roi[1]);
        y_1 = int(roi[0]);
        x_2 = int(roi[3]);
        y_2 = int(roi[2]);
        bbox = (x_1, y_1, x_2-x_1, y_2 - y_1);

        return bbox;

    @classmethod
    def convert_bbox_to_roi(cls, bbox):
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        roi = np.array([p1[1], p1[0], p2[1], p2[0]]);
        return roi;

    @classmethod
    def convert_roi_to_center(cls, roi):

        x_1 = int(roi[1]);
        y_1 = int(roi[0]);
        x_2 = int(roi[3]);
        y_2 = int(roi[2]);

        # Get center of ROI
        pt_image_x = int((x_1 + x_2)/2);
        pt_image_y = int((y_1 + y_2)/2);

        pt_center = (pt_image_x, pt_image_y);

        return pt_center;

    @classmethod
    def find_candidate_roi(cls, tk_end, roi, frame_index, tracker_list, tk_cand_list):
        """Find tracker candidate that match roi from the visual tracker (CRST)

        Args:
            tk_end (TYPE): Original tracker
            roi (TYPE): roi from the visual tracker (CRST)
            frame_index (TYPE): Current frame index
            tracker_list (TYPE): List of tracker
            tk_cand_list (TYPE): List of candidate tracker for this Original tracker

        Returns:
            TYPE: Candidate tracker (matched the roi from visual tracking)
        """

        for tk in tracker_list:
            init_frame_index = tk.get_init_frame_index();
            last_frame_index = tk.get_last_frame_index();


            if init_frame_index <= frame_index and last_frame_index >= frame_index and init_frame_index >= tk_end.get_last_frame_index() - cls.PREVIOUS_HORIZON_CSRT:

                if abs(frame_index - init_frame_index) <= 3 or tk.track_id in tk_cand_list:
                    det_obj_candidiate = tk.get_det_frame_index(frame_index);
                    if det_obj_candidiate is None:
                        # print('[Error]: roi is None');
                        continue;

                    over = det_object.intersection_over_union_rect(roi, det_obj_candidiate.det_2Dbox);

                    if over > cls.OVERLAP_MIN_CRST:
                        # print('tk: {} Init: {} End: {}'.format(tk_end.id, tk_end.init_det_index, tk.last_det_index))
                        # print('tk: {} Init: {} End: {}'.format(tk.id, tk.init_det_index, tk.last_det_index))
                        # print('roi: {} roi_cand: {} over: {}'.format(roi_candidate, roi, over));
                        return tk;


        return None;

    @classmethod
    def plot_rect(cls, frame, bbox, text = None, color = (255,0,0)):

        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, color, 2, 1);

        if not (text is None):
            image = cv2.putText(frame, text, p1, cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    @classmethod
    def merge_track_match(cls, match_list):
        """Merge the list of match pairs into list of track id that correspond to the same track

        Args:
            match_list (TYPE): List of pairs of tracks that should be merged
        """

        new_tk_match_list = [];
        for match in match_list:
            id_end = match[0];
            id_start = match[1];

            new_flag = True;
            for new_tk_post in new_tk_match_list:
                if new_tk_post[-1] == id_end:
                    new_tk_post.append(id_start);
                    new_flag = False;
                    break;

            if new_flag:
                new_tk_match_list.append([id_end, id_start]);


        return new_tk_match_list;

    @classmethod
    def run_merge_tracks(cls, tracker_list, list_img_file, img_folder_path,  det_zone_IM, display = False, debug_track_list=[]):

        match_list = [];

        for break_index, tk in enumerate(list(tracker_list)):


            frame_index = tk.get_last_frame_index();
            det_obj = tk.get_det_frame_index(frame_index);
            if det_obj is None:
                print('[Error]: roi is None');
                continue;

            if len(debug_track_list) > 0:
                if not (tk.track_id in debug_track_list):
                    continue;

            if det_zone_IM.in_zone(det_obj.get_center_det_2Dbox()):
                print('tk: {} end in merge zone'.format(tk.track_id));

                # Start the tracker at the beginning!
                frame_index = tk.get_last_frame_index(nb_past = cls.PREVIOUS_HORIZON_CSRT);
                det_obj = tk.get_det_frame_index(frame_index);

                roi = det_obj.det_2Dbox;

                tracker = cv2.TrackerCSRT_create();

                image_name = list_img_file[frame_index];
                frame = cv2.imread(os.path.join(img_folder_path, image_name));

                bbox = cls.convert_roi_to_bbox(det_obj.det_2Dbox);
                ok = tracker.init(frame, bbox)

                roi_prev = cls.convert_bbox_to_roi(bbox);

                tk_cand_list = [];
                for i in range(cls.HORIZON_CSRT):
                    frame_index +=1;

                    # Break loop if exceed maximum frames
                    if frame_index >= len(list_img_file):
                        break;

                    image_name = list_img_file[frame_index];
                    frame = cv2.imread(os.path.join(img_folder_path, image_name));

                    ok, bbox = tracker.update(frame)
                    if ok:

                        if display:
                            cls.plot_rect(frame, bbox);

                        roi = cls.convert_bbox_to_roi(bbox);

                        # Check if there is continuity int he tracking:
                        over = det_object.intersection_over_union_rect(roi, roi_prev);
                        if over < 0.1:
                            print('[ERROR]: CSRT tracker jumped');
                            break;

                        roi_prev = roi;

                        if not det_zone_IM.in_zone(cls.convert_roi_to_center(roi)):
                            # print('Merge failed: tk: {} out of zone'.format(tk.track_id));
                            break;


                        tk_cand = cls.find_candidate_roi(tk, roi, frame_index, tracker_list, tk_cand_list);
                        if not (tk_cand is None):
                            # print('Merge: tk: {} tk_cand:{}'.format(tk.track_id, tk_cand.id));
                            det_obj_cand = tk_cand.get_det_frame_index(frame_index);
                            if not (det_obj_cand is None):
                                if display:
                                    det_obj_cand.display_on_image(frame, custom_text= 'Id:{}'.format(tk_cand.track_id), color=(0,0,255))
                                    # cls.plot_rect(frame, , cls.convert_roi_to_bbox(roi_cand),  text='Id:{}'.format(tk_cand.track_id), color=(0,0,255));
                                    print('Tk: {} roi: {} tk_cand: {} roi_cand: {} over: {}'.format(tk.track_id, roi, tk_cand.track_id, det_obj_cand.det_2Dbox.transpose(), det_object.intersection_over_union_rect(roi, det_obj_cand.det_2Dbox)));

                            if tk_cand.track_id != tk.track_id:
                                tk_cand_list.append(tk_cand.track_id);

                    else:
                        # Tracking failure
                        if display:
                            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

                        print('Tk id: {} tracker failure'.format(tk.track_id))
                        break;

                    if display:

                        det_zone_IM.display_on_image(frame, thickness = 2)

                        cv2.imshow('frame', frame)
                        if cv2.waitKey(0) & 0xFF == ord('q'):
                            return;


                    if len(tk_cand_list) > cls.MATCHING_THRESH:
                        break;

                # Get candidate:
                if tk_cand_list:
                    tk_cand_id, count = cfgutil.compute_highest_occurence(tk_cand_list);

                    if tk_cand_id != tk.track_id:
                        print('Merge: tk: {} tk_cand:{}'.format(tk.track_id, tk_cand_id));
                        match_list.append([tk.track_id, tk_cand_id])


        # Merge the list of pairs into list of tracks id that should be merged
        tk_match_list = cls.merge_track_match(match_list);

        return tk_match_list;