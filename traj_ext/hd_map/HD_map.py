#################################################################################
#
# HD map
#
#################################################################################

import numpy as np
import os
import sys
import cv2

import collections
import pandas as pd
import argparse

from traj_ext.utils import mathutil
from traj_ext.tracker.cameramodel import CameraModel

class RoadMark(object):
    """Raod Mark structure. Holds a list on points defining a road mark"""
    def __init__(self, road_mark_id, road_mark_type):

        # Unique ID for the road mark to be identified in the HD map
        self.road_mark_id = road_mark_id;

        # Type of road mark
        self.road_mark_type = road_mark_type;

        # List of points defining the road mark
        self.point_list = [];

    def add_point(self, point):
        """Add a point to the road mark

        Args:
            point (TYPE): Point of type np.array shape (2,1)

        Raises:
            ValueError: Point wrong shape
        """
        if point.shape != (2,1):
            raise ValueError('RoadMark point shape: {} !=(2,1)'.format(point.shape));

        self.point_list.append(point);

class HDmap(object):

    """Define a HD map with road mark information

    Attributes:
        COLOR_CURB (TYPE): Description
        COLOR_LANE (TYPE): Description
        COLOR_STOP (TYPE): Description
        TYPE_CURB (TYPE): Description
        TYPE_LANE (TYPE): Description
        TYPE_STOP (TYPE): Description

        name (TYPE): Name of the HD map
        road_marks (TYPE): List of Road Mark
    """

    TYPE_CURB = 100;
    COLOR_CURB = ( 0, 0, 0); #BLACK

    TYPE_LANE_PLAIN = 200;
    COLOR_LANE_PLAIN = (255, 0, 0); #BLUE

    TYPE_LANE_DOTTED = 201;
    COLOR_LANE_DOTTED = (255, 0, 0); #BLUE

    TYPE_STOP_PLAIN = 300;
    COLOR_STOP_PLAIN = (0, 0, 255); #RED

    TYPE_STOP_DOTTED = 301;
    COLOR_STOP_DOTTED = (0, 0, 255); #RED

    """Hold information about the agent type of a track"""
    def __init__(self, name, origin_latlon = np.array([0., 0.])):

        self.name = name;
        self.road_marks = [];

        self.origin_latlon = origin_latlon;

    def get_color(self, road_mark_type):
        """Get the color for a road mark type

        Args:
            road_mark_type (TYPE): Type of road mark

        Returns:
            TYPE: Color tulpe
        """

        color = ( 0, 0, 0);
        if road_mark_type == self.TYPE_CURB:
            color = self.COLOR_CURB;

        elif road_mark_type == self.TYPE_LANE_PLAIN:
            color = self.COLOR_LANE_PLAIN;

        elif road_mark_type == self.TYPE_LANE_DOTTED:
            color = self.COLOR_LANE_DOTTED;

        elif road_mark_type == self.TYPE_STOP_PLAIN:
            color = self.COLOR_STOP_PLAIN;

        elif road_mark_type == self.TYPE_STOP_DOTTED:
            color = self.COLOR_STOP_DOTTED;

        else:
            print('[WARNING] Road mark type {} unknown'.format(road_mark_type));

        return color;

    def get_road_mark(self, road_mark_id):
        """Get Road Mark by id

        Args:
            road_mark_id (TYPE): Id of the road mark

        Returns:
            TYPE: RoadMark
        """
        road_mark_result = None;
        for road_mark in self.road_marks:
            if road_mark.road_mark_id == road_mark_id:
                road_mark_result = road_mark;

        return road_mark_result;

    def delete_last_road_mark(self, road_mark_id):
        """Delete the last point of the last road mark
        """

        road_mark = self.get_road_mark(road_mark_id);

        if not (road_mark is None):

            if len(road_mark.point_list) > 0:
                road_mark.point_list.pop();

            # if len(road_mark.point_list) == 0:
            #     self.road_marks.pop();

    def set_origin_latlon(self, origin_latitude, origin_longitude):
        """Set origin Lat/Lon of the local frame used in the HD map

        Args:
            origin_latitude (TYPE): Description
            origin_longitude (TYPE): Description
        """

        print('Setting origin: lat:{} lon:{}'.format(origin_latitude, origin_longitude))
        self.origin_latlon[0] = origin_latitude;
        self.origin_latlon[1] = origin_longitude;


    def add_xy_offset(self, x_offset, y_offset):

        print('Add offset: x:{} y:{}'.format(x_offset, y_offset))
        for road_mark in self.road_marks:

            for point in road_mark.point_list:

                point[0] += x_offset;
                point[1] += y_offset;


    def add_point(self, road_mark_id, road_mark_type , point):
        """Add a point to a specific road mark. Create road mark if necessary

        Args:
            road_mark_id (TYPE): Unique RoadMark Id
            road_mark_type (TYPE): Type of road mark
            point (TYPE): Point ot add of shape (2,1)

        """
        road_mark = self.get_road_mark(road_mark_id);

        if road_mark is None:
            road_mark = RoadMark(road_mark_id, road_mark_type);
            self.road_marks.append(road_mark);

        if road_mark.road_mark_type != road_mark_type:
            self.set_road_mark_type(road_mark_id, road_mark_type);

        road_mark.add_point(point);

    def set_road_mark_type(self, road_mark_id, road_mark_type):
        """Set the road mark type for a specific road mark

        Args:
            road_mark_id (TYPE): Description
            road_mark_type (TYPE): Description
        """
        road_mark = self.get_road_mark(road_mark_id);

        if not (road_mark is None):

            print('Changing road_mark_type: {}'.format(road_mark_type));
            road_mark.road_mark_type = road_mark_type;

    def get_road_mark_type(self, road_mark_id):
        """Get the road mark type for a specific road mark

        Args:
            road_mark_id (TYPE): Description
        """
        road_mark = self.get_road_mark(road_mark_id);

        if not (road_mark is None):

            # print('Changing road_mark_type: {}'.format(road_mark_type));
            road_mark_type = road_mark.road_mark_type;

        return road_mark_type;

    def display_on_image(self, image, cam_model, show_number = False, thickness = 1):
        """Display the HD map on an image

        Args:
            image (TYPE): Image
            cam_model (TYPE): Camera Model associated with the image

        Returns:
            TYPE: Image
        """
        for road_mark in self.road_marks:

            if len(road_mark.point_list) > 0:
                color = self.get_color(road_mark.road_mark_type);

                pt_previous = None;
                for point in road_mark.point_list:

                    # Project point:
                    point_31 = np.append(point, np.array([[0]]), axis=0);
                    pt_current = cam_model.project_points(point_31);

                    # # Project 3Dbox corners on Image Plane
                    # point_31.shape = (1,3);
                    # (pt_img, jacobian) = cv2.projectPoints(point_31, cam_model.rot_CF_F, cam_model.trans_CF_F, cam_model.cam_matrix, cam_model.dist_coeffs)

                    # # Cretae list of tulpe
                    # pt_current = (int(pt_img[0][0][0]), int(pt_img[0][0][1]));

                    # Draw circle
                    cv2.circle(image, (pt_current[0], pt_current[1]),2, color, -1);

                    # Draw line
                    if not (pt_previous is None):
                        cv2.line(image, (pt_previous[0],pt_previous[1]), (pt_current[0],pt_current[1]), color, thickness = thickness)

                    pt_previous = pt_current;

                if show_number:
                    cv2.putText(image, str(road_mark.road_mark_id), (pt_current[0],pt_current[1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)

        return image;

    def create_view(self, im_size = (800, 800), dist_max = 0.0):
        """Create a satellite view of the HD map. Usefull to plot trajectories.

        Args:
            im_size (tuple, optional): Size of the image

        Returns:
            TYPE: CameraModel, Image
        """

        focal_lenght = max(im_size);
        half_size_pix =  max(im_size)/2;

        if dist_max < 0.1:
            for road_mark in self.road_marks:
                for point in road_mark.point_list:
                    if max(abs(point[0]),abs(point[1])) > dist_max:
                        dist_max = max(abs(point[0]),abs(point[1]));

        z_cam = float(dist_max *focal_lenght)/float(half_size_pix);

        # Create camera parameters
        rot_CF_F = np.identity(3);
        trans_CF_F = np.array([0.0,0.0,z_cam]);
        trans_CF_F.shape = (3,1)

        center = (im_size[1]/2, im_size[0]/2)
        camera_matrix = np.array(
                                 [[focal_lenght, 0, center[0]],
                                 [0, focal_lenght, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )

        dist_coeffs = np.zeros((4,1))

        # Create camera model
        cam_model = CameraModel(rot_CF_F, trans_CF_F, camera_matrix, dist_coeffs);

        # Create blank image
        image = 255*np.ones((im_size[0],im_size[1],3), np.uint8);

        return cam_model, image;

    def to_csv(self, path_to_csv):

        dict_pd = collections.OrderedDict.fromkeys(['point_type',\
                                                     'point_id',\
                                                     'x_ned',\
                                                     'y_ned',\
                                                     'latitude',\
                                                     'longitude',\
                                                     'origin_latitude',\
                                                     'origin_longitude']);


        dict_pd['point_type'] = [];
        dict_pd['point_id'] = [];
        dict_pd['x_ned'] = [];
        dict_pd['y_ned'] = [];
        dict_pd['latitude'] = [];
        dict_pd['longitude'] = [];
        dict_pd['origin_latitude'] = [];
        dict_pd['origin_longitude'] = [];

        for road_mark in self.road_marks:

            for point_id, point_ned in enumerate(road_mark.point_list):

                dict_pd['point_type'].append(road_mark.road_mark_type);
                dict_pd['point_id'].append(point_id);
                dict_pd['x_ned'].append(point_ned[0,0]);
                dict_pd['y_ned'].append(point_ned[1,0]);

                # Compute latlon of point
                latlon_21 = mathutil.NED_to_latlon(self.origin_latlon, point_ned);

                dict_pd['latitude'].append(latlon_21[0]);
                dict_pd['longitude'].append(latlon_21[1]);

                dict_pd['origin_latitude'].append(self.origin_latlon[0]);
                dict_pd['origin_longitude'].append(self.origin_latlon[1]);

        # Create dataframe
        df = pd.DataFrame(dict_pd);

        # Write dataframe in csv
        print('Saving hd_map:{} to {}'.format(self.name, path_to_csv))
        df.to_csv(path_to_csv, index=False);

    @classmethod
    def from_csv(self, path_to_csv):

        # Create HD map:
        if not os.path.isfile(path_to_csv):
            print('[HDmap]: HD map file does not exist: {}'.format(path_to_csv));
            return None;

        # Extract name
        name = path_to_csv.split('/')[-1].split('.')[0];

        # Create HD map
        hd_map = HDmap(name);

        # Read dataframe with panda
        df = pd.read_csv(path_to_csv);

        # Group by type
        grouped = df.groupby(['point_type'], sort=True);

#        hd_map.set_origin_latlon(52.104081, 23.786570);

        # Iterate trhough type:
        origin_latlon_flag = True;
        road_mark_id = -1; # define unique ID per instance of Road Mark
        for point_type, rows in grouped:

            for index, row in rows.iterrows():

                # Set Origin Lat/lon
                if origin_latlon_flag:
                    lat = float(row['origin_latitude']);
                    lon = float(row['origin_longitude']);
                    hd_map.set_origin_latlon(lat, lon);
                    origin_latlon_flag = False;

                # Increment road mark id when point_type goes back to 0
                if row['point_id'] == 0:
                    road_mark_id +=1;

                # Get road mark type
                road_mark_type = int(row['point_type']);

                # Convert lat/lon in NED point
                pos_ned_21 = np.array([row['x_ned'], row['y_ned']]);

                pos_ned_21.shape = (2,1);

                # Add points
                hd_map.add_point(road_mark_id, road_mark_type, pos_ned_21);

        return hd_map;



    @classmethod
    def from_csv_berkeley(cls, path_to_csv):
        """Create HD map from txt file

        Args:
            path_to_csv (TYPE): Path to the file

        Returns:
            TYPE: HDmap object
        """

        # TEMPORARY

        # Defining origin for Varna and Brest:
        # ORIGIN_BREST_LAT = 52.104081;
        # ORIGIN_BREST_LON = 23.786570;

        ORIGIN_BREST_LAT = 52.10409834618578;
        ORIGIN_BREST_LON = 23.78652442003926;

        origin_latlon = np.array([ORIGIN_BREST_LAT, ORIGIN_BREST_LON]);

        # Create HD map:
        if not os.path.isfile(path_to_csv):
            print('[HDmap]: HD map file does not exist: {}'.format(path_to_csv));
            return None;

        # Extract name
        name = path_to_csv.split('/')[-1].split('.')[0];

        # Create HD map
        hd_map = HDmap(name);

        # Read dataframe with panda
        df = pd.read_csv(path_to_csv, delim_whitespace=True);

        # Group by type
        grouped = df.groupby(['point_type'], sort=True);

        # Iterate trhough type:
        road_mark_id = -1; # define unique ID per instance of Road Mark
        for point_type, rows in grouped:

            for index, row in rows.iterrows():

                # Increment road mark id when point_type goes back to 0
                if row['point_id'] == 0:
                    road_mark_id +=1;

                # Get road mark type
                road_mark_type = int(row['point_type']/100);

                # Convert lat/lon in NED point
                latlon_21 = np.array([row['latitude'], row['longitude']]);

                pos_ned_21 = mathutil.latlon_to_NED(origin_latlon, latlon_21);
                pos_ned_21.shape = (2,1);

                # Add points
                hd_map.add_point(road_mark_id, road_mark_type, pos_ned_21);

        return hd_map;

if __name__ == '__main__':

    # Arg parser:
    parser = argparse.ArgumentParser(description='Draw HD maps on image');
    parser.add_argument('-hd_map', dest="hd_map_path", type=str, help='Path of the HD map file', default='traj_ext/camera_calib/calib_file/brest/brest_area1_street_hd_map.csv');
    parser.add_argument('-image', dest="image_path", type=str, help='Path of the image', default='traj_ext/camera_calib/calib_file/brest/brest_area1_street.jpg');
    parser.add_argument('-camera', dest="cam_model_path", type=str, help='Path of the satellite camera model yml', default='traj_ext/camera_calib/calib_file/brest/brest_area1_street_cfg.yml');
    args = parser.parse_args();

    # Create HD map
    # hd_map = HDmap.from_csv_berkeley(args.hd_map_path);
    hd_map = HDmap.from_csv(args.hd_map_path);
    # hd_map.to_csv('brest_berkeley_hdmap.csv')

    # Open camera model
    if os.path.isfile(args.cam_model_path) and os.path.isfile(args.image_path):
        # Open camera model
        cam_model = CameraModel.read_from_yml(args.cam_model_path);

        # Open image
        image = cv2.imread(args.image_path);

    # Create image and camera for top-down view
    else:
        cam_model, image = hd_map.create_view();

    image = hd_map.display_on_image(image, cam_model);

    cv2.imshow('HD map', image);

    key = cv2.waitKey(0) & 0xFF
