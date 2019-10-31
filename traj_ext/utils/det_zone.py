# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-03-21 14:12:39
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

import csv
import numpy as np
import cv2

class DetZoneFNED(object):

    """Object that holds a detection zone in FNED frame (world local frame)"""
    def __init__(self, pt_det_zone_FNED):

        if not (pt_det_zone_FNED.shape[1] == 3):
            raise ValueError('Wrong dimension for pt_det_zone_FNED {}'.format(pt_det_zone_FNED));

        self.pt_det_zone_FNED = pt_det_zone_FNED;

    @classmethod
    def read_from_yml(cls, input_path):
        """Read Detection Zone from yml config file

        Args:
            input_path (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            ValueError: Wrong dimension for the points defining the detection zone
        """
        # Check input path
        if input_path == '':
            raise ValueError('[Error] Detection Zone F input path empty: {}'.format(input_path));

        fs_read = cv2.FileStorage(input_path, cv2.FILE_STORAGE_READ)
        pt_det_zone_FNED = fs_read.getNode('model_points_FNED').mat();

        det_zone_FNED = DetZoneFNED(pt_det_zone_FNED);

        return det_zone_FNED;

    def save_to_yml(self, output_path):
        """Save the Detection zone to yml file

        Args:
            output_path (TYPE): Description
        """
        # Write Param in YAML file
        fs_write = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
        fs_write.write('model_points_FNED', self.pt_det_zone_FNED);
        fs_write.release()
        print('\nDetection Zone F config file saved %s' %(output_path));

    def display_on_image(self, image, cam_model, color=(0,0,255), thickness = 1):
        """Display the dection zone on an image

        Args:
            image (TYPE): Image
            cam_model (TYPE): Camera Model for the image
            color (tuple, optional): Color
            thickness (int, optional): Thickness

        Returns:
            TYPE: image
        """

        # Create Image det zone:
        det_zone_im = self.create_det_zone_image(cam_model);

        # Draw on image
        image = det_zone_im.display_on_image(image, color=color, thickness=thickness);

        return image;

    def create_det_zone_image(self, cam_model):
        """Create Detection Zone Image from Detction Zone FNED

        Args:
            image (TYPE): Image
            cam_model (TYPE): Camera Model corresponding to the image

        Returns:
            TYPE: Detection Zone Image
        """
        # Project point on image plane
        pt_img_list = cam_model.project_list_pt_F(self.pt_det_zone_FNED);

        #Convert tulpe into numpy array
        pt_img_np = np.array([], np.int32);
        pt_img_np.shape = (0,2);
        for pt_img_tulpe in pt_img_list:

            daz = np.array([pt_img_tulpe[0], pt_img_tulpe[1]], np.int32);
            daz.shape = (1,2);
            pt_img_np = np.append(pt_img_np, daz, axis=0);


        # Create Image det zone:
        det_zone_im = DetZoneImage(pt_img_np);

        return det_zone_im;


    def in_zone(self, pt_FNED):
        """Check if the point is in the Detection zone or not

        Args:
            pt (TYPE): Poitn FNED shape (3,1)

        Returns:
            TYPE: Boolean True if point is in the detection zone
        """
        in_zone = True;
        if self.pt_det_zone_FNED is not None:
            if (pt_FNED.shape[0] < 2):
                print('[Error]: Point FNED not right shape: {}'.format(pt_FNED.shape));
                in_zone = True;
                return in_zone;


            # Detection Zone from yml
            contour = self.pt_det_zone_FNED.astype(int);
            contour = contour[:,0:2]

            pt = (pt_FNED[0], pt_FNED[1]);
            res = cv2.pointPolygonTest(contour, pt, False)
            if (res < 1):
                in_zone = False;

        return in_zone;

    def shrink_zone(self, shrink_factor = 0.8):
        """Create new zone from the current zone by applying a shrink factor

        Args:
            shrink_factor (float, optional): Shrink factor

        Returns:
            TYPE: Detection Zone FNED
        """
        center = self.pt_det_zone_FNED.mean(axis = 0);

        pt_det_zone_FNED_shrink = center + shrink_factor*(self.pt_det_zone_FNED - center);

        det_zone_shrink_FNED = DetZoneFNED(pt_det_zone_FNED_shrink);

        return det_zone_shrink_FNED;


class DetZoneImage(object):
    """Object that holds a detection zone in Image frame"""

    def __init__(self, pt_det_zone_IM):

        if not (pt_det_zone_IM.shape[1] == 2):
            raise ValueError('Wrong dimension for pt_det_zone_IM {}'.format(pt_det_zone_IM));

        self.pt_det_zone_IM = pt_det_zone_IM;

    def display_on_image(self, image, color=(0,0,255), thickness = 1):
        """Display the detection zone on the image

        Args:
            image (TYPE): Image
            color (tuple, optional): Description
            thickness (int, optional): Description

        Returns:
            TYPE: Image
        """

        pt_img_np = self.pt_det_zone_IM.reshape((-1,1,2))
        pt_img_np = pt_img_np.astype(int);

        cv2.polylines(image,[pt_img_np],True,color, thickness=thickness)

        return image;

    @classmethod
    def read_from_yml(cls, input_path):
        """Read Detection Zone from yml config file

        Args:
            input_path (TYPE): Path to the yml config file

        Returns:
            TYPE: DetZoneImage object

        Raises:
            ValueError: Wrong path for the config file
        """

        # Check input path
        if input_path == '':
            raise ValueError('[Error] Detection Zone IM input path empty: {}'.format(input_path));

        fs_read = cv2.FileStorage(input_path, cv2.FILE_STORAGE_READ)
        pt_det_zone_IM = fs_read.getNode('model_points_IM').mat();

        # Create object
        det_zone_IM = DetZoneImage(pt_det_zone_IM);

        return det_zone_IM;

    def save_to_yml(self, output_path):
        """Save the Detection zone to yml file

        Args:
            output_path (TYPE): Path
        """

        # Write Param in YAML file
        fs_write_im = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
        fs_write_im.write('model_points_IM', self.pt_det_zone_IM);
        fs_write_im.release()
        print('\nDetection Zone IM config file saved %s' %(output_path));


    def in_zone(self, pt):
        """Check if the point is in the Detection zone or not

        Args:
            pt (TYPE): Point (x,y)

        Returns:
            TYPE: Boolean True if point is in the detection zone
        """
        in_zone = True;
        if not (self.pt_det_zone_IM is None):

            # Detection Zone from yml
            contour = self.pt_det_zone_IM.astype(int);
            contour = contour[:,0:2]

            # TO DO: Change this to better test
            res = cv2.pointPolygonTest(contour, pt, False)
            if res < 1:
                in_zone = bool(False)

        return in_zone;


    def shrink_zone(self, shrink_factor = 0.8):
        """Create new zone from the current zone by applying a shrink factor

        Args:
            shrink_factor (float, optional): Shrink factor

        Returns:
            TYPE: Detection Zone FNED
        """
        center = self.pt_det_zone_IM.mean(axis = 0);

        pt_det_zone_IM_shrink = center + shrink_factor*(self.pt_det_zone_IM - center);

        det_zone_shrink_IM = DetZoneImage(pt_det_zone_IM_shrink);

        return det_zone_shrink_IM;

