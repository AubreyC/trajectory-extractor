#################################################################################
#
# Track in image plane, made of 2D detections
#
#################################################################################

import numpy as np
import os
import sys
import cv2
import csv
import pandas as pd
import collections

from traj_ext.utils import mathutil
from traj_ext.utils import det_zone
from traj_ext.object_det.det_object import DetObject

from traj_ext.box3D_fitting.box3D_object import Box3DObject

from traj_ext.utils import cfgutil

class DetAssociation():
    def __init__(self, frame_index, track_id, det_id):
        self.track_id = track_id;
        self.det_id = det_id;
        self.frame_index = frame_index;

class MeasObject(object):
    """docstring for MeasObject"""
    def __init__(self, frame_index, det_obj, box3D = None):

        self.frame_index = frame_index;
        self.det_obj = det_obj;
        self.box3D = box3D;

class Track2D(object):

    """Track of 2D detections"""
    def __init__(self, track_id, agent_type = None, color = None, delete_past_mask = 0):

        self.track_id = track_id;
        self.color = color;
        if (self.color is None):
            self.color = (int(np.random.randint(0,255,1)[0]), int(np.random.randint(0,255,1)[0]), int(np.random.randint(0,255,1)[0]));

        self._det_list = [];
        self.agent_type = agent_type;

        self._init_frame_index = None;
        self._last_frame_index = None;

        self._delete_past_mask = delete_past_mask;

    @classmethod
    def get_track2D_by_id(cls, track_2D_list, track_id):
        """Get track2D from a track2D list with track_id

        Args:
            track_2D_list (TYPE): List of track2D
            track_id (TYPE): Track ID

        Returns:
            TYPE: track2D
        """
        result = None;
        for track_2D in track_2D_list:
            if track_2D.track_id == track_id:
                result = track_2D;
                break;

        return result;

    @classmethod
    def export_det_asso_csv(cls, list_img_name, track_2D_list, det_asso_folder_path):
        """Export list of track2D in csv, with one csv by frame

        Args:
            list_img_name (TYPE): Description
            track_2D_list (TYPE): Description
            det_asso_folder_path (TYPE): Description
        """
        print('Track2D: Saving detection association in: {}'.format(det_asso_folder_path));

        total_frame_index = len(list_img_name);
        for frame_index, image_name in enumerate(list_img_name):

            det_asso_list = [];
            for track_2D in track_2D_list:

                det = track_2D.get_det_frame_index(frame_index);

                if not (det is None):
                    det_asso = DetAssociation(frame_index, track_2D.track_id, det.det_id);

                    det_asso_list.append(det_asso);



            det_asso_csv_name = image_name.split('.')[0] + '_detassociation.csv';
            det_asso_path = os.path.join(det_asso_folder_path, det_asso_csv_name);

            cls.write_det_asso_csv(det_asso_list, det_asso_path);

            cfgutil.progress_bar(frame_index, total_frame_index, 'Saving detection association')

    @classmethod
    def from_csv(cls, list_img_name, det_folder_path, det_asso_folder_path, box3D_folder = '', expand_mask = False, frame_limit = 0):
        """Read Track2D from detection and detection associations files

        Args:
            list_img_name (TYPE): List of image name (used to get the corresponding det and det asso)
            det_folder_path (TYPE): Folder for detections files
            det_asso_folder_path (TYPE): Folder for detection association file

        Returns:
            TYPE: List of track 2D
        """

        track_2D_list = [];
        frame_index_list = [];

        # Iterate over det/det_asso files
        for frame_index, image_name in enumerate(list_img_name):

            # Break at frame limit
            if frame_limit > 0 and frame_index > frame_limit:
                print("[Track 2D]: Read csv reached frame limit: {}\n".format(frame_limit));
                break;

            # Create frame index list
            frame_index_list.append(frame_index);

            # Get detection association at frame_index
            det_asso_csv_name = image_name.split('.')[0] + '_detassociation.csv';
            det_asso_path = os.path.join(det_asso_folder_path, det_asso_csv_name);

            det_asso_list = cls.read_det_asso_csv(det_asso_path);

            # Get detection at frame_index
            det_csv_name = image_name.split('.')[0] + '_det.csv';
            det_path = os.path.join(det_folder_path, det_csv_name);

            det_list = DetObject.from_csv(det_path, expand_mask = expand_mask);

            # Display status
            status_str = 'Reading track2D: {}/{}'.format(frame_index, len(list_img_name));
            cfgutil.progress_bar(frame_index, len(list_img_name), status_str);
            # print('frame:{} det_csv_name:{} det_asso_csv_name:{}'.format(frame_index, det_csv_name, det_asso_csv_name))

            # Get box3D if available:
            box3D_list = [];
            if os.path.isdir(box3D_folder):
                # Get detection at frame_index
                box3D_csv_name = image_name.split('.')[0] + '_3Dbox.csv';
                box3D_path = os.path.join(box3D_folder, box3D_csv_name);

                box3D_list = Box3DObject.from_csv(box3D_path);

            # Iterate over det_asso
            for det_asso in det_asso_list:

                # Get corresponding det object
                det = DetObject.get_det_from_id(det_list, det_asso.det_id);

                # Only add the detection to the track if the detection is good
                if det.good:

                    # Get corresponding det object box3D if available
                    box3D = Box3DObject.get_box3D_from_id(box3D_list, det_asso.det_id);

                    # Get track_2d for this track_id or create one if necessary
                    track_2D = cls.get_track2D_by_id(track_2D_list, det_asso.track_id);
                    if track_2D is None:
                        track_2D = Track2D(det_asso.track_id);
                        track_2D_list.append(track_2D);

                    # Push detection to corresponding track_2d
                    track_2D.push_det(frame_index, det, box3D);

        # Set the agent type from highest detection occurence
        for track_2D in track_2D_list:
            track_2D.set_agent_type_highest();

        print('');
        return track_2D_list, frame_index_list;

    @classmethod
    def read_det_asso_csv(cls, csv_path):
        """Read detection association file. A file per frame, composed of det_id and track_id.

        Args:
            csv_path (TYPE): Path to the detection association csv file

        Returns:
            TYPE: List of detection association
        """

        # Create empty list of detection_asso object
        det_asso_list = [];

        try:
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)

                # Iterate over row
                for row in reader:

                    # To ensure compatibilty with previous csv without frame_index
                    frame_index = 0;
                    if frame_index in row.keys():
                        frame_index = int(row['frame_index']);

                    track_id = int(row['track_id']);
                    det_id = int(row['det_id']);

                    # Create det_asso object
                    det_asso = DetAssociation(frame_index, track_id, det_id);

                    # Append to the list
                    det_asso_list.append(det_asso);

        except FileNotFoundError as e:
            print('[WARNING]: read_det_asso_csv.from_csv() could not open file: {}'.format(e))

        return det_asso_list;

    @classmethod
    def write_det_asso_csv(cls, det_asso_list, csv_path):
        """Write detection assoaciation file. A file per frame, composed of det_id and track_id.

        Args:
            det_asso_list (TYPE): List of detection association
            csv_path (TYPE): Path to the csv file
        """


        dict_data = collections.OrderedDict.fromkeys(['frame_index', 'det_id', 'track_id']);

        dict_data['frame_index'] = [];
        dict_data['det_id'] = [];
        dict_data['track_id'] = [];

        for det_asso in det_asso_list:

            dict_data['frame_index'].append(det_asso.frame_index);
            dict_data['det_id'].append(det_asso.det_id);
            dict_data['track_id'].append(det_asso.track_id);

        # Create dataframe
        df = pd.DataFrame(dict_data);

        # Sort by track_id:
        df.sort_values(by=['det_id'], inplace = True);

        # Write dataframe in csv
        df.to_csv(csv_path, index=False);

    def merge_with_track2D(self, new_track_2D):
        """Add measurement from new track_2D to current track 2D.

        Args:
            new_track_2D (TYPE): New track 2D
        """

        for frame_index in range(new_track_2D.get_init_frame_index(), new_track_2D.get_last_frame_index() + 1):

            meas = new_track_2D.get_meas_frame_index(frame_index);

            if not (meas is None):
                self.push_det(frame_index, meas.det_obj, meas.box3D);

    def set_agent_type(self, agent_type):
        """Set the agent_type

        Args:
            agent_type (TYPE): Description
        """

        self.agent_type = agent_type;

    def set_agent_type_highest(self):
        """Set the agent_type according to the most frequent detedction label

        Returns:
            TYPE: Description
        """
        label_list = [];
        for meas in self._det_list:
            if not (meas is None):
                label_list.append(meas.det_obj.label);

        max_label, count = cfgutil.compute_highest_occurence(label_list);

        self.set_agent_type(max_label);

        return max_label;


    def push_det(self, frame_index, det_object, box3D = None):
        """Push detection object in the track 2D.

        Args:
            frame_index (TYPE): Frame index
            det_object (TYPE): Detection object
            box3D (None, optional): Box3D

        Raises:
            ValueError: Description
        """

        if not (det_object is None):

            # If init
            if self._init_frame_index is None:
                self._init_frame_index = frame_index;

                # Put the track_id in the det
                det_object.track_id = self.track_id;

                if det_object.frame_id != frame_index:
                    raise ValueError('[Warning]: Track2D frame_index:{} != det_object.frame_id: {}'.format(frame_index, det_object.frame_id));

                self._last_frame_index = frame_index;

                meas = MeasObject(frame_index, det_object, box3D);
                self._det_list.append(meas);


            # If in between frame
            elif frame_index >= self._init_frame_index and frame_index <= self._last_frame_index:

                print('[Track 2D]: Replacing det object at: {}'.format(frame_index));

                # Put the track_id in the det
                det_object.track_id = self.track_id;

                if det_object.frame_id != frame_index:
                    raise ValueError('[Warning]: Track2D frame_index:{} != det_object.frame_id: {}'.format(frame_index, det_object.frame_id));

                index = self.get_index_det_frame_index(frame_index);
                if not (index is None):
                    meas = MeasObject(frame_index, det_object, box3D);
                    self._det_list[index] = meas;

            elif frame_index > self._last_frame_index:

                # Add None for all frame in between
                for i in range(self._last_frame_index, frame_index - 1):
                    self._det_list.append(None);

                # Put the track_id in the det
                det_object.track_id = self.track_id;

                if det_object.frame_id != frame_index:
                    raise ValueError('[Warning]: Track2D frame_index:{} != det_object.frame_id: {}'.format(frame_index, det_object.frame_id));

                self._last_frame_index = frame_index;

                meas = MeasObject(frame_index, det_object, box3D);
                self._det_list.append(meas);

            elif frame_index < self._init_frame_index:

                # Add None for all frame in between
                for i in range(frame_index, self._init_frame_index - 1):
                    self._det_list.insert(0, None);

                # Put the track_id in the det
                det_object.track_id = self.track_id;

                if det_object.frame_id != frame_index:
                    raise ValueError('[Warning]: Track2D frame_index:{} != det_object.frame_id: {}'.format(frame_index, det_object.frame_id));

                self._init_frame_index = frame_index;

                meas = MeasObject(frame_index, det_object, box3D);
                self._det_list.insert(0, meas);

            # Remove the mask of past detections to avoid memory explosion
            if self._delete_past_mask > 0:
                if len(self._det_list) > self._delete_past_mask:
                    meas = self._det_list[-self._delete_past_mask-1];
                    if not ( meas is None):
                        meas.det_obj.remove_mask(no_mask_array = True);

        else:
            print('[ERROR]: det_object None for: frame_index:{} track_id:{}'.format(frame_index, self.track_id))

    def get_last_agent_type(self):
        """Return the label of the last detection

        Returns:
            TYPE: Agent type of the last detection

        Raises:
            ValueError: No detection in the list
        """

        if len(self._det_list) < 1:
            raise ValueError('Track2D self._det_list empty')

        return self._det_list[-1].det_obj.label;

    def get_init_frame_index(self):
        """Return the frame index of the beginning of the track

        Returns:
            TYPE: Init frame index
        """
        if len(self._det_list) < 1:
            raise ValueError('Track2D self._det_list empty')

        # return self._det_list[0].frame_id;
        return self._init_frame_index;



    def get_last_frame_index(self, nb_past = 0):
        """Return the frame index of the end of the track

        Returns:
            TYPE: Last frame index

        Args:
            nb_past (int, optional): Get the frame index of the nb_past detection form the last one

        Raises:
            ValueError: Det list is empty
        """

        if len(self._det_list) < 1:
            raise ValueError('Track2D self._det_list empty')

        # index = self._det_list[-1].frame_id;
        index = self._last_frame_index;

        if nb_past > 0:

            count = 0;
            for meas in reversed(self._det_list):
                if not (meas is None):

                    if count == nb_past:
                        index = meas.det_obj.frame_id;
                        break;

                    count += 1;

        return index;

    def get_length(self):
        """Return the lenght of the track in frame number

        Returns:
            TYPE: Lenght in number of frame
        """
        return len(self._det_list);

    def get_index_det_frame_index(self, frame_index):

        if len(self._det_list) < 1:
            raise ValueError('Track2D self._det_list empty')

        # Method to avoid iterating over det_list: push None in det_list when no detection
        # access directly with index

        index_result = None;
        index = frame_index - self.get_init_frame_index();
        if index >= 0 and index < len(self._det_list):
            index_result = index;

        return index_result;

    def display_on_image(self, frame_index, image, cam_model, color_text = (0, 0, 255), no_label = False):

        meas = self.get_meas_frame_index(frame_index);
        if not (meas is None):

            det = meas.det_obj;
            box3D = meas.box3D;

            if not (det is None):
                det.display_on_image(image, color = self.color, color_text = color_text, no_label = no_label);

            if not (box3D is None):
                if not (cam_model is None):
                    box3D.display_on_image(image, cam_model, color = self.color);

        return;

    def get_meas_frame_index(self, frame_index):
        """Return measurement object at specific frame index

        Args:
            frame_index (TYPE): Frame index

        Returns:
            TYPE: Meas object
        """

        meas_result = None;

        # Get index of det from frame_index
        index = self.get_index_det_frame_index(frame_index);
        if not (index is None):
            meas_result = self._det_list[index];

        return meas_result;

    def get_det_frame_index(self, frame_index):
        """Return the detection object at a specific frame index

        Args:
            frame_index (TYPE): Frame index

        Returns:
            TYPE: Detection object
        """

        det_object_result = None;

        # Get index of det from frame_index
        meas = self.get_meas_frame_index(frame_index);
        if not (meas is None):
            det_object_result =  meas.det_obj;

        if not (det_object_result is None):
            if frame_index != det_object_result.frame_id:
                    print('[WARNING]: {} != {}'.format(frame_index, det_object_result.frame_id))

        return det_object_result;

