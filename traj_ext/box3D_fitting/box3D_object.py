# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-03-21 14:12:39
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

import csv
import numpy as np
import cv2
import copy
import pandas as pd
import collections

from traj_ext.object_det import det_object

from traj_ext.utils import mathutil

class Type3DBoxStruct():
    def __init__(self, label, length, width, height):
        self.label = label;
        self.length = float(length);
        self.width = float(width);
        self.height = -abs(float(height));
        self.box3D_lwh = [ self.length, self.width, self.height];


    @classmethod
    def write_box3D_type_csv(cls, path_csv, type_3DBox_list):
        csv_open = False;
        with open(path_csv, 'w') as csvfile:

            fieldnames = [];

            # Add new keys
            fieldnames.append('label');
            fieldnames.append('length');
            fieldnames.append('width');
            fieldnames.append('height');

            #Write field name
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
            writer.writeheader();

            for type_3DBox in type_3DBox_list:

                # Field management
                dict_row = {};

                dict_row['label'] = type_3DBox.label;
                dict_row['length'] = type_3DBox.length;
                dict_row['width'] = type_3DBox.width;
                dict_row['height'] = type_3DBox.height;

                # Write detection in CSV
                writer.writerow(dict_row);

    @classmethod
    def read_type_csv(cls, csv_path):

        data_list = [];

        # Read CSV
        with open(csv_path) as csvfile:
            # Open CSV sa a dict
            reader = csv.DictReader(csvfile)

            # Read each row
            for row in reader:

                fields = row.keys();

                if 'label' in fields:
                    label = row['label'];

                if 'width' in fields:
                    width = np.float64(row['width']);

                if 'height' in fields:
                    height = np.float64(row['height']);

                if 'length' in fields:
                    length = np.float64(row['length']);

                # Create data struct
                type_data  = Type3DBoxStruct(label, length, width, height);
                data_list.append(type_data);

        return data_list

    @classmethod
    def default_3DBox_list(cls):
        """Create default type Box3D for car, trcuk, bus, person, bicycle, motorcycle

        Returns:
            TYPE: List of default Type box3D
        """

        # Define default 3D box type:
        default_type_3DBox_list = [ Type3DBoxStruct('car', 4.0, 1.8, 1.6),\
                                    Type3DBoxStruct('bus', 12.0, 2.6, 2.5),\
                                    Type3DBoxStruct('truck', 6.0, 2.4, 2.0),\
                                    Type3DBoxStruct('person', 0.8, 0.8, 1.6),\
                                    Type3DBoxStruct('motorcycle', 1.2, 0.8, 1.6),\
                                    Type3DBoxStruct('bicycle', 1.2, 0.8, 1.6)];

        return default_type_3DBox_list;


class Box3DObject(object):

    """Object that holds a detection on an image"""
    def __init__(self, psi_rad, x, y, z, length, width, height, det_id = 0, percent_overlap = 1.0):

        self.psi_rad = psi_rad;         # Orientation of the box (yaw) - rad
        self.x = x;                     # Position X in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
        self.y = y;                     # Position Y in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
        self.z = z;                     # Position Z in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
        self.length = length;           # Length of the 3D Box - meters
        self.width = width;             # Width of the 3D Box - meters
        self.height = height;           # Height of the 3D Box - meters

        self.percent_overlap = percent_overlap; # Use for estimating 3D box from 2D mask
        self.det_id = det_id;

    def set_det_id(self, det_id):
        """Set Detection Id

        Args:
            det_id (TYPE): det id
        """
        self.det_id = det_id;

    def set_percent_overlap(self, percent_overlap):
        """Set percentage overlap

        Args:
            percent_overlap (TYPE): Description

        Returns:
            TYPE: Description
        """

        self.percent_overlap = percent_overlap;
        return;

    @classmethod
    def from_csv(cls, csv_path):
        """Read 3D box object from csv

        Args:
            csv_path (TYPE): Path to the csv

        Returns:
            TYPE: Description
        """

        box3D_list = [];

        # Create dict

        try:
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:

                    box3D = cls.from_dict(row);
                    box3D_list.append(box3D);

        except FileNotFoundError as e:
            print('[WARNING]: Box3DObject.from_csv() could not open file: {}'.format(e))

        return box3D_list;


    @classmethod
    def from_dict(self, dict_data):
        """Create 3D box object from dict

        Args:
            dict_data (TYPE): Dict

        Returns:
            TYPE: 3D box object
        """

        fields = dict_data.keys();

        if 'det_id' in fields:
            det_id = int(dict_data['det_id']);

        if 'percent_overlap' in fields:
            percent_overlap = float(dict_data['percent_overlap']);

        if 'box_3D_psi' in fields:
            psi_rad = float(dict_data['box_3D_psi']);

        if 'box_3D_x' in fields:
            x = float(dict_data['box_3D_x']);

        if 'box_3D_y' in fields:
            y = float(dict_data['box_3D_y']);

        if 'box_3D_z' in fields:
            z = float(dict_data['box_3D_z']);

        if 'box_3D_l' in fields:
            l = float(dict_data['box_3D_l']);

        if 'box_3D_w' in fields:
            w = float(dict_data['box_3D_w']);

        if 'box_3D_h' in fields:
            h = float(dict_data['box_3D_h']);

        box3D = Box3DObject(psi_rad, x, y, z, l, w, h, det_id = det_id, percent_overlap = percent_overlap);

        return box3D;


    @classmethod
    def remove_box3D_percentoverlap(box3D_list, threshold):
        """Remove 3D box that are under the overlap threshold
           Overlap between 3D box and mask

        Args:
            data_box3D_list (list): Description
            threshold (list): Description

        Returns:
            TYPE: Description
        """

        for box3D in list(box3D_list):

            if not box3D.check_percentoverlap(threshold):
                data_box3D_list.remove(box3D);

        return box3D_list;

    @classmethod
    def get_box3D_from_id(cls, box3D_list, det_id):
        """Get box3D from det_id

        Args:
            box3D_list (TYPE): List of Box3D object
            det_id (TYPE): Detection ID

        Returns:
            TYPE: Box3D object

        """
        result = None;
        for box3D in box3D_list:

            if box3D.det_id == det_id:
                result = box3D;

        return result;

    def check_percentoverlap(self, threshold):
        """Check if percent_overlap is above threshold

        Args:
            threshold (TYPE): Description

        Returns:
            TYPE: Description
        """

        return self.percent_overlap >= threshold;

    @classmethod
    def to_csv(cls, csv_path, box3D_list):
        """Write 3D box list to a csv file

        Args:
            csv_path (TYPE): Path to the csv
            box3D_list (TYPE): List of 3Dbox object

        Returns:
            TYPE: csv_path
        """
        # Create list of dict
        dict_data_list = [];
        for box3D in box3D_list:
            dict_data_list.append(box3D.to_dict());

        if len(dict_data_list)>0:

            # Create dataframe
            df_det = pd.DataFrame(dict_data_list);

            # Sort by track_id:
            df_det.sort_values(by=['det_id'], inplace = True);

            # Write dataframe in csv
            df_det.to_csv(csv_path, index = False);

        return csv_path;

    def to_dict(self):
        """Create a dict with 3D box data

        Returns:
            TYPE: Dict
        """

        dict_data = collections.OrderedDict.fromkeys(['det_id', 'percent_overlap', 'box_3D_psi', 'box_3D_x', 'box_3D_y', 'box_3D_z', 'box_3D_l', 'box_3D_w', 'box_3D_h']);

        dict_data['det_id'] = self.det_id;
        dict_data['percent_overlap'] = self.percent_overlap;

        dict_data['box_3D_psi'] = self.psi_rad;
        dict_data['box_3D_x'] = self.x;
        dict_data['box_3D_y'] = self.y;
        dict_data['box_3D_z'] = self.z;
        dict_data['box_3D_l'] = self.length;
        dict_data['box_3D_w'] = self.width;
        dict_data['box_3D_h'] = self.height;

        return dict_data;

    def create_3Dbox(self):
        """Create 3Dbox corners points in NED frame from the 3D box paramters

        Returns:
            TYPE: List of 3D points in NED frame
        """

        # Get the position of the center of the bottom side:
        c = np.array([self.x, self.y, self.z]);
        c.shape = (3,1)

        # Frame F attached to to box:
        # Frame F is defined by the euler angles: Phi, Theta, Psi (roll, pitch, Yaw)
        eulers_F_NED = [0.0, 0.0, self.psi_rad];
        # Rotation matrix: Transfom vector from frame F1 to F2: X_F2 = R_F2_F1*X_F1
        rot_F_NED = mathutil.eulerAnglesToRotationMatrix(eulers_F_NED);
        rot_NED_F = rot_F_NED.transpose();

        # Generate the Corner points:
        tr = np.array([self.length/2, -self.width/2, 0]);
        tr.shape = (3,1)
        pt1 = c + rot_NED_F.dot(tr);

        tr = np.array([self.length/2, self.width/2, 0]);
        tr.shape = (3,1)
        pt2 = c + rot_NED_F.dot(tr);

        tr = np.array([-self.length/2, -self.width/2, 0]);
        tr.shape = (3,1)
        pt3 = c + rot_NED_F.dot(tr);

        tr = np.array([-self.length/2, self.width/2, 0]);
        tr.shape = (3,1)
        pt4 = c + rot_NED_F.dot(tr);

        tr = np.array([self.length/2, -self.width/2, self.height]);
        tr.shape = (3,1)
        pt5 = c + rot_NED_F.dot(tr);

        tr = np.array([self.length/2, self.width/2, self.height]);
        tr.shape = (3,1)
        pt6 = c + rot_NED_F.dot(tr);

        tr = np.array([-self.length/2, -self.width/2, self.height]);
        tr.shape = (3,1)
        pt7 = c + rot_NED_F.dot(tr);

        tr = np.array([-self.length/2, self.width/2, self.height]);
        tr.shape = (3,1)
        pt8 = c + rot_NED_F.dot(tr);

        # Create list from corner points
        list_pt = [pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8];
        return list_pt;

    def project_box_bottom_center_image(self, cam_model):
        """Project the center of the bottom side of the 3D box on an image

        Args:
            cam_model (TYPE): Camera model

        Returns:
            TYPE: Point on image
        """

        pos_FNED = np.array([self.x, self.y, self.z]);
        pos_FNED.shape = (3,1)

        # Create list from corner points
        pix_meas = cam_model.project_points(pos_FNED);
        pix_meas.shape = (2,1);

        return pix_meas;

    def project_box_image(self, cam_model):
        """Project the corners of the 3D box on an image

        Args:
            cam_model (TYPE): Camera model

        Returns:
            TYPE: List of point on image
        """

        list_pt_F = self.create_3Dbox();

        list_pt_im = cam_model.project_list_pt_F(list_pt_F);

        return list_pt_im;

    def create_mask(self, cam_model, im_size):
        """Create a mask of the projected 3D box

        Args:
            cam_model (TYPE): Description

        Returns:
            TYPE: Description
        """

        # Project point on image plane
        pt_img_list = self.project_box_image(cam_model);

        # Create mask:
        mask = det_object.create_mask_image(im_size, pt_img_list);

        return mask;

    def display_mask_on_image(self, image, cam_model, color = (255,0,0)):
        """Display 3D box projected mask on image

        Args:
            image (TYPE): Image
            cam_model (TYPE): Camera model
            color (tuple, optional): Color of the mask

        Returns:
            TYPE: Image
        """
        mask = self.create_mask(cam_model);

        image = det_object.draw_mask(image, cam_model, color = color);

        return image;

    # Draw the 3D Box on the Image Plane (Image from the camera)
    def display_on_image(self, image, cam_model, color = (255,0,0), thickness = 2, color_front = (255,255,0)):
        """DIsplay 3D box on image

        Args:
            image (TYPE): Image
            cam_model (TYPE): Camera model
            color (tuple, optional): Color of the 3D box
            thickness (int, optional): Thickness of the 3D box
            color_front (tuple, optional): Color of the front of the 3D box

        Returns:
            TYPE: Image with 3D box drawn on it
        """

        if not (image is None):
            # Project point on image plane
            pt_img_list = self.project_box_image(cam_model);

            for pt_img_tulpe in pt_img_list:

                # Draw Center of the bottom side point on image
                cv2.circle(image, pt_img_tulpe, 2, color, -1)

            # Draw line to form the box
            cv2.line(image, pt_img_list[0], pt_img_list[1], color_front, thickness)
            cv2.line(image, pt_img_list[0], pt_img_list[2], color, thickness)
            cv2.line(image, pt_img_list[1], pt_img_list[3], color, thickness)
            cv2.line(image, pt_img_list[2], pt_img_list[3], color, thickness)

            cv2.line(image, pt_img_list[4], pt_img_list[5], color_front, thickness)
            cv2.line(image, pt_img_list[4], pt_img_list[6], color, thickness)
            cv2.line(image, pt_img_list[5], pt_img_list[7], color, thickness)
            cv2.line(image, pt_img_list[6], pt_img_list[7], color, thickness)

            cv2.line(image, pt_img_list[0], pt_img_list[4], color_front, thickness)
            cv2.line(image, pt_img_list[1], pt_img_list[5], color_front, thickness)
            cv2.line(image, pt_img_list[2], pt_img_list[6], color, thickness)
            cv2.line(image, pt_img_list[3], pt_img_list[7], color, thickness)

        return image;

    def get_projected_2Dbox(self, cam_model):
        """Get the projected 2D box from the 3D box

        Args:
            cam_model (TYPE): Camera model

        Returns:
            TYPE: Projected 2D box
        """
        pt_img_list = self.project_box_image(cam_model);
        pt_img_list = np.array(pt_img_list)

        rect_xywh = cv2.boundingRect(pt_img_list);
        box2D = np.array([rect_xywh[1], rect_xywh[0], rect_xywh[1] + rect_xywh[3], rect_xywh[0] + rect_xywh[2]]);
        box2D.shape = (4,1);

        return box2D;