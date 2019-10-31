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

def scale_mask(scale_x, scale_y, mask):
    """ Scale the Mask according to the scaling factor used for the image

    Args:
        scale_x (TYPE): Scaling factor x
        scale_y (TYPE): Scaling factor y
        mask (TYPE): Description

    Returns:
        TYPE: Description
    """

    mask = mask.astype('uint8');
    mask = cv2.resize(mask,None,fx=scale_x, fy=scale_y, interpolation = cv2.INTER_CUBIC)
    mask = mask.astype(np.bool);

    return mask;

def scale_2Dbox(scale_x, scale_y, box2D):
    """ Scale the box2D according to the scaling factor used for the image

    Args:
        scale_x (TYPE): Scaling factor x
        scale_y (TYPE): Scaling factor y
        box2D (TYPE): box2D [y1, x1, y2, x2]

    Returns:
        TYPE: Description
    """

    box2D[0]  = box2D[0]*scale_y;
    box2D[1]  = box2D[1]*scale_x;
    box2D[2]  = box2D[2]*scale_y;
    box2D[3]  = box2D[3]*scale_x;

    return box2D


def draw_mask(image, mask, color = (255,0,0)):
    """Display boolean mask on image

    Args:
        image (TYPE): Image
        mask (TYPE): Boolean mask
        color (TYPE): Color of the mask

    Returns:
        TYPE: Description

    """
    clrImg = np.zeros(image.shape, image.dtype)

    clrImg[:,:] = color;

    m = np.array(mask, dtype = "uint8")
    clrMask = cv2.bitwise_and(clrImg, clrImg, mask=m);

    cv2.addWeighted(image, 1.0, clrMask, 0.5, 0.0, image)

    return image;

def create_mask_image(image_size, pt_img_list):
    """Create a mask from list of points

    Args:
        im_size (TYPE): Image size (tuple)
        pt_img_list (TYPE): List of points on image

    Returns:
        TYPE: Boolean mask
    """

    #Convert tulpe into numpy array
    pt_img_np = np.array([], np.int32);
    pt_img_np.shape = (0,2);
    for pt_img_tulpe in pt_img_list:

        daz = np.array([pt_img_tulpe[0], pt_img_tulpe[1]], np.int32);
        daz.shape = (1,2);
        pt_img_np = np.append(pt_img_np, daz, axis=0);


    #Find Convex Hull from the rectangles points
    hull = cv2.convexHull(pt_img_np)

    mask = np.zeros((image_size[0], image_size[1]), np.int8);
    cv2.fillConvexPoly(mask, hull, 1, lineType=8, shift=0);

    mask = mask.astype(np.bool);

    return mask;

def intersection_over_union_mask(mask_1, mask_2):
    """Compute the intersection over union between masks

    Args:
        mask_1 (TYPE): Description
        mask_2 (TYPE): Description

    Returns:
        TYPE: Intersection over union
    """

    # Overlap Mask
    overlap_mask = np.logical_and(mask_1, mask_2);

    # Union mask
    union_mask = np.logical_or(mask_1, mask_2);

    # Count the 1
    count_overlap =  np.count_nonzero(overlap_mask);
    count_union =  np.count_nonzero(union_mask);

    percent_overlap =  float(count_overlap) / float(count_union);

    return percent_overlap;

def intersection_over_union_rect(rect_1, rect_2):
    """Compute the intersection over union between rect

    Args:
        rect_1 (TYPE): Rect shape (4,1)
        rect_2 (TYPE): Rect shape (4,1)

    Returns:
        TYPE: intersection over union
    """
    over = intersection_rect(rect_1, rect_2);

    union = intersection_rect(rect_1, rect_1) +  intersection_rect(rect_2, rect_2) - over;

    result = 0;
    if float(union) > 0.01:
        result = float(over)/float(union);

    return result;

def intersection_rect(rect_1, rect_2):
    """Compute the intersection area betwen two rectangle

    Args:
        rect_1 (TYPE): Rect shape (4,1)
        rect_2 (TYPE): Rect shape (4,1)

    Returns:
        TYPE: Intersection area
    """

    # print('rect_1: {}'.format(type(rect_1)))
    # print('rect_2: {}'.format(type(rect_2)))

    def get_overlap_segment(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]));

    result = get_overlap_segment(rect_1[[0,2]], rect_2[[0,2]]) * get_overlap_segment(rect_1[[1,3]], rect_2[[1,3]]);
    return result;


def encode_mask(mask, width, height):

    # Encode mask by just taking the contour of the masK
    # One dimension

    m = copy.copy(mask);

    # One dimension array
    m.shape = (1, m.size)

    # Padding: Add 0 to the left and right of the array (in order to be able to substarct shifted version)
    og_mask = np.append(np.zeros((1,1), int), m)
    og_mask = np.append(og_mask, np.zeros((1,1), int))

    # Starting
    og_mask_shifted_left = np.append(m, np.zeros((1,2), int))
    diff_start = np.subtract(og_mask, og_mask_shifted_left)
    # Shift in index due to padding cancel with the fact that the -1 appears at (index_start - 1)
    diff_index_start = np.where(diff_start == -1)[0];

    # Ending
    og_mask_shifted_right = np.append(np.zeros((1,2), int), m )
    diff_end = np.subtract(og_mask, og_mask_shifted_right)
    diff_index_end = np.where(diff_end == -1)[0];
    # Compensate for padding and the fact the -1 appears at (index_end + 1)
    diff_index_end = diff_index_end -2;

    diff_index = np.append(diff_index_start, diff_index_end);
    diff_index = np.sort(diff_index);
    return diff_index

def decode_mask_bool(diff_index, width, height):
    mask_new = np.zeros((1, width*height), np.bool);

    length = int(diff_index.size/2)
    for i in range(0, length):

        be = diff_index[2*i]
        end = diff_index[2*i+1]
        mask_new[0,be:end+1] = True;

    mask_new.shape = (height, width);
    return mask_new

def convert_str(diff_mask):
    b = ''.join(str(x) + ' ' for x in diff_mask)
    return  b.strip();

def back_str(b_str):
    a = b_str.split(' ')
    myarray = np.asarray([], 'int')
    if len(a) > 1:
        mask_bin = [int(x) for x in a];
        myarray = np.asarray(mask_bin, 'int')
    return myarray

class DetObject(object):

    """Object that holds a detection on an image"""
    def __init__(self, det_id, label, det_2Dbox, confidence, track_id = None, good = True, mask_array = None, image_width = None, image_height = None, det_mask = None, frame_name = None, frame_id = None):

        self.label = label;

        if not (det_2Dbox.shape == (4,1)):
            raise ValueError('Wrong size for det_2Dbox {}'.format(det_2Dbox));
        self.det_2Dbox = det_2Dbox;
        self.det_mask = det_mask;

        self.frame_id = frame_id;
        self.det_id = det_id;

        self.track_id = track_id;

        self.confidence = confidence;

        self.mask_array = mask_array;
        self.image_width = image_width;
        self.image_height = image_height;

        self.frame_name = frame_name;

        self.good = good;

    @classmethod
    def get_max_det_id(cls, det_object_list):
        """Get maximum id from a det_object list

        Args:
            det_object_list (TYPE): List of det object

        Returns:
            TYPE: Id
        """
        max_id = -1;
        for det_o in det_object_list:

            if det_o.det_id > max_id:
                max_id = det_o.det_id;

        return max_id;

    @classmethod
    def get_det_from_id(cls, det_object_list, det_id):
        """Get detection from det_id

        Args:
            det_object_list (TYPE): List of detection object
            det_id (TYPE): Detection ID

        Returns:
            TYPE: Detection object
        """
        result = None;
        for det_o in det_object_list:

            if det_o.det_id == det_id:
                result = det_o;

        return result;


    @classmethod
    def from_csv(cls, csv_path, expand_mask = False):
        """Read detection from detection csv

        Args:
            csv_path (TYPE): Path to the csv
            expand_mask (bool, optional): Expand the mask

        Returns:
            TYPE: Description
        """

        det_object_list = [];

        # Create dict
        try:
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:

                    det_object = cls.from_dict(row, expand_mask = expand_mask);
                    det_object_list.append(det_object);

        except FileNotFoundError as e:
            print('[WARNING]: DetOject.from_csv() could not open file: {}'.format(e))

        return det_object_list;

    @classmethod
    def to_csv(cls, csv_path, det_object_list):
        """Write a list of detection object in a csv file

        Args:
            csv_path (TYPE): Description
            det_object_list (TYPE): Description

        Returns:
            TYPE: Description
        """
        # Create list of dict
        dict_data_list = [];
        for det_object in det_object_list:
            dict_data_list.append(det_object.to_dict());

        if len(dict_data_list)>0:

            # Create dataframe
            df_det = pd.DataFrame(dict_data_list);

            # Sort by track_id:
            df_det.sort_values(by=['det_id'], inplace = True);

            # Write dataframe in csv
            df_det.to_csv(csv_path, index = False);

        return;

    @classmethod
    def from_mask(cls, det_id, label, mask, confidence, frame_name = '', frame_id = None):
        """Create a detection object from a mask. Used when creating a detection object manually.

        Args:
            det_id (TYPE): ID
            label (TYPE): Label
            mask (TYPE): Mask
            confidence (TYPE): Confidence level
            frame_name (str, optional): Name of the frame
            frame_id (None, optional): Frame ID

        Returns:
            TYPE: Description
        """
        i, j = np.where(mask)
        x_1 = min(j);
        x_2 = max(j);
        y_1 = min(i);
        y_2 = max(i);

        width = mask.shape[1];
        height = mask.shape[0];

        det_2Dbox = np.array([y_1, x_1, y_2, x_2], dtype= np.int16);
        det_2Dbox.shape = (4,1);

        # Create list of dict
        det_object = DetObject(det_id, label, det_2Dbox, confidence, image_width = width, image_height = height, det_mask = mask, frame_name = frame_name, frame_id = frame_id)

        return det_object;


    @classmethod
    def from_dict(self, dict_data, expand_mask = False):
        """Create detection object from a dict with corresponding fields

        Args:
            dict_data (TYPE): Dict
            expand_mask (bool, optional): Expand mask flag

        Returns:
            TYPE: Deteciton object
        """

        fields = dict_data.keys();

        frame_name = None;
        if 'frame_name' in fields:
            frame_name = dict_data['frame_name'];

        frame_id = None;
        if 'frame_id' in fields:
            if dict_data['frame_id'] == '':
                frame_id = None;
            else:
                frame_id = int(float(dict_data['frame_id']));

        if 'det_id' in fields:
            det_id = int(dict_data['det_id']);

        if 'width' in fields:
            image_width = int(dict_data['width']);

        if 'height' in fields:
            image_height = int(dict_data['height']);

        if 'confidence' in fields:
            confidence = np.float32(dict_data['confidence']);

        if 'topleft_y' in fields:
            det_2Dbox = np.array([dict_data['topleft_y'], dict_data['topleft_x'], dict_data['bottomright_y'], dict_data['bottomright_x']], dtype= np.int16);
            det_2Dbox.shape = (4,1);

        if 'label' in fields:
            label = dict_data['label'];

        if 'mask' in fields:
            mask_str = dict_data['mask'];
            mask_array = back_str(mask_str);

        good = True; # TEMP
        if 'good' in fields:
            good = bool(int(dict_data['good']));

        det_object = DetObject(det_id, label, det_2Dbox, confidence, good = good, frame_id = frame_id, frame_name = frame_name, mask_array = mask_array, image_width = image_width, image_height = image_height);

        # Expend mask
        if expand_mask:
            det_object.expand_mask();

        return det_object;

    def to_dict(self):
        """Create a dictionnary with detection object data

        Returns:
            TYPE: Dict
        """
        dict_data = collections.OrderedDict.fromkeys(['det_id', 'frame_name', 'frame_id', 'label', 'good', 'width', 'height', 'confidence', 'topleft_y', 'topleft_x', 'bottomright_y', 'bottomright_x', 'mask']);

        dict_data['det_id'] = self.det_id;
        dict_data['frame_name'] = self.frame_name;
        dict_data['frame_id'] = self.frame_id;

        dict_data['label'] = self.label;
        dict_data['good'] = int(self.good);

        dict_data['width'] = self.image_width;
        dict_data['height'] = self.image_height;
        dict_data['confidence'] = self.confidence;

        dict_data['topleft_y'] = self.det_2Dbox[0,0];
        dict_data['topleft_x'] = self.det_2Dbox[1,0];
        dict_data['bottomright_y'] = self.det_2Dbox[2,0];
        dict_data['bottomright_x'] = self.det_2Dbox[3,0];

        if not (self.mask_array is None):
            dict_data['mask'] = convert_str(self.mask_array);

        else:
            dict_data['mask'] = convert_str(encode_mask(self.det_mask, self.image_width, self.image_height));

        return dict_data;

    def expand_mask(self):
        """Tranform mask_array to a mask as a boolean array of the size of the image

        Returns:
            TYPE: Success boolean
        """
        success = False;
        if not (self.mask_array is None):
            det_mask = decode_mask_bool(self.mask_array, self.image_width, self.image_height);
            det_mask.shape = (self.image_height, self.image_width);

            self.det_mask = det_mask;
            success = True;

        return success;

    def remove_mask(self, no_mask_array = False):
        """Remove the full mask from the object to save memory

        Args:
            no_mask_array (bool, optional): Do not compute and save the mask as array

        Returns:
            TYPE: Description

        """

        if not (self.det_mask is None):

            # Encode the mask into mask array
            if self.mask_array is None and (not no_mask_array):
                self.mask_array = encode_mask(self.det_mask, self.image_width, self.image_height);

            # Delete the mask
            self.det_mask = None;

        return;

    def get_center_det_2Dbox(self):
        """Compute the center of the 2D detection box

        Returns:
            TYPE: Description
        """
        # Get ROI coordinates
        x_1 = int(self.det_2Dbox[1]);
        y_1 = int(self.det_2Dbox[0]);
        x_2 = int(self.det_2Dbox[3]);
        y_2 = int(self.det_2Dbox[2]);

        # Get center of ROI
        pt_image_x = int((x_1 + x_2)/2);
        pt_image_y = int((y_1 + y_2)/2);

        pt_center = (pt_image_x, pt_image_y);

        return pt_center;

    def display_on_image(self, image, color = None, color_text = (0, 0, 255), no_label = False, no_2Dbox = False, no_mask = False, custom_text = None, track_id_text=False):
        """Display the detection on an image

        Args:
            image (TYPE): Image to display the detection on
            color (None, optional): Specific color for the 2D box and mask
            no_label (bool, optional): Disable label
            no_2Dbox (bool, optional): Disable 2D box
            no_mask (bool, optional): Disable mask

        Returns:
            TYPE: Void
        """

        # Define color
        if color is None:
            color = (np.random.randint( low = 0, high = 255), np.random.randint( low = 0, high = 255), np.random.randint( low = 0, high = 255));

        if not (self.det_2Dbox is None):

            x_1 = int(self.det_2Dbox[1]);
            y_1 = int(self.det_2Dbox[0]);
            x_2 = int(self.det_2Dbox[3]);
            y_2 = int(self.det_2Dbox[2]);

            tl = (x_1, y_1)
            br = (x_2, y_2)

            # Display 2D bounding box
            if not no_2Dbox:
                image = cv2.rectangle(image, tl, br, color, 1);

                if not self.good:
                    image = cv2.line(image, tl, br, (0,0,255), 2);

            # Display custom text
            text = '';
            if custom_text:
                text = custom_text;

            elif track_id_text:
                track_id = '';
                if not (self.track_id is None):
                    track_id = self.track_id;
                text = 'D:{} T:{} {}'.format(self.det_id, track_id, self.label)

            # Display label
            elif not no_label:
                text = '{} {}'.format(self.det_id, self.label)

            image = cv2.putText(image, text, self.get_center_det_2Dbox(), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_text, 1)

        # Display mask
        if not (self.det_mask is None):

            clrImg = np.zeros(image.shape, image.dtype)
            clrImg[:,:] = color;

            m = np.array(self.det_mask, dtype = "uint8")
            clrMask = cv2.bitwise_and(clrImg, clrImg, mask=m);

            cv2.addWeighted(image, 1.0, clrMask, 0.5, 0.0, image)

        return image;

    def is_point_in_det_2Dbox(self, pt_x, pt_y):
        """Check if point is in the 2D box of the detection object

        Args:
            pt_x (TYPE): X coordinate
            pt_y (TYPE): X coordinate

        Returns:
            TYPE: Boolean
        """

        x_1 = int(self.det_2Dbox[1]);
        y_1 = int(self.det_2Dbox[0]);
        x_2 = int(self.det_2Dbox[3]);
        y_2 = int(self.det_2Dbox[2]);

        result = pt_x > x_1 and pt_x < x_2 and \
                 pt_y > y_1 and pt_y < y_2;

        return result;

    def to_scale(self, scale_x, scale_y):
        """Convert a detection object to a scaled version of this detection object

        Args:
            scale (TYPE): Scale factor
        """

        new_det_2Dbox = copy.copy(self.det_2Dbox);
        new_det_2Dbox = scale_2Dbox(scale_x, scale_y, new_det_2Dbox);

        new_det_mask = copy.copy(self.det_mask);
        new_det_mask = scale_mask(scale_x, scale_y, new_det_mask);

        new_det_object = DetObject(self.det_id, self.label, new_det_2Dbox, self.confidence, image_width = int(scale_x*self.image_width), image_height = int(scale_y*self.image_height), det_mask = new_det_mask, frame_name = self.frame_name, frame_id = self.frame_id);

        return new_det_object;

    def from_cropped_image(self, x1, y1, x2, y2, image_width, image_height):
        """Convert a detection object that comes from a cropped image
           into the corresponding detection object for the orignal image.

        Args:
            x1 (TYPE): x1 cropped region
            y1 (TYPE): y1 cropped region
            x2 (TYPE): x2 cropped region
            y2 (TYPE): y2 cropped region
            image_width (TYPE): Original image width
            image_height (TYPE): Original image height

        Returns:
            TYPE: Description
        """

        # New det box:
        new_det_2Dbox = np.zeros((4,1), dtype= np.int16);

        new_det_2Dbox[1] = self.det_2Dbox[1] + x1;
        new_det_2Dbox[0] = self.det_2Dbox[0] + y1;
        new_det_2Dbox[3] = self.det_2Dbox[3] + x1;
        new_det_2Dbox[2] = self.det_2Dbox[2] + y1;

        # New mask:
        new_det_mask = np.zeros((image_height, image_width), np.bool);

        new_det_mask[y1:y2, x1:x2] = self.det_mask;

        det_object = DetObject(self.det_id, self.label, new_det_2Dbox, self.confidence, image_width = image_width, image_height = image_height, det_mask = new_det_mask, frame_name = self.frame_name, frame_id = self.frame_id);

        return det_object;