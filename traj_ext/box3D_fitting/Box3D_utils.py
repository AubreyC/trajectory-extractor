# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-07-24 10:44:20
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:


###################################################################################
#
# 3DBox_utils.py defines a bunch of usefull function to compute the projection /
# reprojection of 3D box on the Image Plane (camera view).
# This is usefull to compute the 3D position of an object from it's mask.
#
###################################################################################

import numpy as np
import cv2
import sys
import scipy.optimize as opt


from utils.mathutil import *

# Draw a Mask on the image:
# Mask is a bool array of the dim of the image
def draw_mask(im, mask, color):
    clrImg = np.zeros(im.shape, im.dtype)

    clrImg[:,:] = color;

    m = np.array(mask, dtype = "uint8")
    clrMask = cv2.bitwise_and(clrImg, clrImg, mask=m);

    cv2.addWeighted(im, 1.0, clrMask, 0.5, 0.0, im)

    return im;

# Create the corners points of the 3D box from 3D box parameters (Yaw, Position, Size)
def create_3Dbox(param_3Dbox):

    # Get box paramaters
    phi = float(param_3Dbox[0]); # Orientation of the box (yaw) - degrees
    x = float(param_3Dbox[1]);   # Position X in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
    y = float(param_3Dbox[2]);   # Position Y in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
    z = float(param_3Dbox[3]);   # Position Z in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
    l = float(param_3Dbox[4]);   # Length of the 3D Box - meters
    w = float(param_3Dbox[5]);   # Width of the 3D Box - meters
    h = float(param_3Dbox[6]);   # Height of the 3D Box - meters

    # Get the position of the center of the bottom side:
    c = np.array([x, y, z]);
    c.shape = (3,1)

    # Frame F attached to to box:
    # Frame F is defined by the euler angles: Phi, Theta, Psi (roll, pitch, Yaw)
    eulers_F_NED = [np.deg2rad(0), np.deg2rad(0) , np.deg2rad(phi)];
    # Rotation matrix: Transfom vector from frame F1 to F2: X_F2 = R_F2_F1*X_F1
    rot_F_NED = eulerAnglesToRotationMatrix(eulers_F_NED);
    rot_NED_F = rot_F_NED.transpose();

    # Generate the Corner points:
    tr = np.array([l/2, -w/2, 0]);
    tr.shape = (3,1)
    pt1 = c + rot_NED_F.dot(tr);

    tr = np.array([l/2, w/2, 0]);
    tr.shape = (3,1)
    pt2 = c + rot_NED_F.dot(tr);

    tr = np.array([-l/2, -w/2, 0]);
    tr.shape = (3,1)
    pt3 = c + rot_NED_F.dot(tr);

    tr = np.array([-l/2, w/2, 0]);
    tr.shape = (3,1)
    pt4 = c + rot_NED_F.dot(tr);

    tr = np.array([l/2, -w/2, h]);
    tr.shape = (3,1)
    pt5 = c + rot_NED_F.dot(tr);

    tr = np.array([l/2, w/2, h]);
    tr.shape = (3,1)
    pt6 = c + rot_NED_F.dot(tr);

    tr = np.array([-l/2, -w/2, h]);
    tr.shape = (3,1)
    pt7 = c + rot_NED_F.dot(tr);

    tr = np.array([-l/2, w/2, h]);
    tr.shape = (3,1)
    pt8 = c + rot_NED_F.dot(tr);

    # Create list from corner points
    list_pt = [pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8];
    return list_pt;

# Project the corners points of the 3D box on the Image Plane
# Usefull to get a mask of the 3D Box
def project_box_image(cam_model, list_pt_F):

    pt_img_list = [];
    for pt in list_pt_F:
            # Project 3Dbox corners on Image Plane
            pt.shape = (1,3);
            (pt_img, jacobian) = cv2.projectPoints(pt, cam_model.rot_CF_F, cam_model.trans_CF_F, cam_model.cam_matrix, cam_model.dist_coeffs)

            # Cretae list of tulpe
            pt_img_tulpe = (int(pt_img[0][0][0]), int(pt_img[0][0][1]));
            pt_img_list.append(pt_img_tulpe);


    return pt_img_list;

# Draw the 3D Box on the Image Plane (Image from the camera)
def draw_3Dbox(im_current, cam_model, list_pt_F, color = (255,0,0), thickness = 2):

    if not (im_current is None):
        # Project point on image plane
        pt_img_list = project_box_image(cam_model, list_pt_F);
        color_front = (255,255,0);

        for pt_img_tulpe in pt_img_list:

            # Draw Center of the bottom side point on image
            cv2.circle(im_current, pt_img_tulpe, 2, color, -1)

        # Draw line to form the box
        cv2.line(im_current, pt_img_list[0], pt_img_list[1], color_front, thickness)
        cv2.line(im_current, pt_img_list[0], pt_img_list[2], color, thickness)
        cv2.line(im_current, pt_img_list[1], pt_img_list[3], color, thickness)
        cv2.line(im_current, pt_img_list[2], pt_img_list[3], color, thickness)

        cv2.line(im_current, pt_img_list[4], pt_img_list[5], color_front, thickness)
        cv2.line(im_current, pt_img_list[4], pt_img_list[6], color, thickness)
        cv2.line(im_current, pt_img_list[5], pt_img_list[7], color, thickness)
        cv2.line(im_current, pt_img_list[6], pt_img_list[7], color, thickness)

        cv2.line(im_current, pt_img_list[0], pt_img_list[4], color_front, thickness)
        cv2.line(im_current, pt_img_list[1], pt_img_list[5], color_front, thickness)
        cv2.line(im_current, pt_img_list[2], pt_img_list[6], color, thickness)
        cv2.line(im_current, pt_img_list[3], pt_img_list[7], color, thickness)

    return im_current;

# Create the mask of the 3D Box on the Image Plane
def create_mask(im_size, cam_model, list_pt_F):

    # Project point on image plane
    pt_img_list = project_box_image(cam_model, list_pt_F);

    #Convert tulpe into numpy array
    pt_img_np = np.array([], np.int32);
    pt_img_np.shape = (0,2);
    for pt_img_tulpe in pt_img_list:

        daz = np.array([pt_img_tulpe[0], pt_img_tulpe[1]], np.int32);
        daz.shape = (1,2);
        pt_img_np = np.append(pt_img_np, daz, axis=0);


    #Find Convex Hull from the rectangles points
    hull = cv2.convexHull(pt_img_np)

    mask = np.zeros((im_size[0], im_size[1]), np.int8);
    cv2.fillConvexPoly(mask, hull, 1, lineType=8, shift=0);

    mask = mask.astype(np.bool);

    return mask;


def draw_det_zone(img, cam_model, list_pt_F, color=(0,0,255), thickness = 1):

    # Project point on image plane
    pt_img_list = project_box_image(cam_model, list_pt_F);

    #Convert tulpe into numpy array
    pt_img_np = np.array([], np.int32);
    pt_img_np.shape = (0,2);
    for pt_img_tulpe in pt_img_list:

        daz = np.array([pt_img_tulpe[0], pt_img_tulpe[1]], np.int32);
        daz.shape = (1,2);
        pt_img_np = np.append(pt_img_np, daz, axis=0);


    pt_img_np = pt_img_np.reshape((-1,1,2))
    cv2.polylines(img,[pt_img_np],True,color, thickness=thickness)

    return ;

def in_detection_zone(roi, pt_det_zone_pix):
    """Limit the filtering to the detection within a zone by keeping only objects present within the zone

    Args:
        data_box3D_list (list)
        pt_det_zone_FNED (bool)

    Returns:
        list
    """
    # Detection Zone: from DET_ZONE_F_PATH or basic_detection_zone_flag (hard coded rectangle)
    in_zone = True;

    if pt_det_zone_pix is not None or basic_detection_zone_flag:

        # Get ROI coordinates
        x_1 = int(roi[1]);
        y_1 = int(roi[0]);
        x_2 = int(roi[3]);
        y_2 = int(roi[2]);

        tl = (x_1, y_1)
        br = (x_2, y_2)

        # Get center of ROI
        pt_image_x = (x_1 + x_2)/2
        pt_image_y = (y_1 + y_2)/2

        pt = (pt_image_x, pt_image_y)

        if pt_det_zone_pix is not None:
            # Detection Zone from yml
            contour = pt_det_zone_pix.astype(int);
            contour = contour[:,0:2]

            # TO DO: Change this to better test
            res = cv2.pointPolygonTest(contour, pt, False)
            if res < 1:
                in_zone = bool(False)

    return in_zone

def pt_det_zone_FNED_to_pix(pt_det_zone_FNED, cam_model):

    pt_det_zone_pix = np.zeros((pt_det_zone_FNED.shape[0], 2), np.int32)

    for idx, pt in enumerate(pt_det_zone_FNED):

        # Project on Image
        pt_F = np.array(pt_det_zone_FNED[idx, :]);
        pt_F.shape = (3,1)

        (pt_img, jacobian) = cv2.projectPoints(pt_F.transpose(), cam_model.rot_CF_F, cam_model.trans_CF_F, cam_model.cam_matrix, cam_model.dist_coeffs)
        pt_img_tulpe = (int(pt_img[0][0][0]), int(pt_img[0][0][1]));

        pix_meas = np.array([int(pt_img[0][0][0]), int(pt_img[0][0][1])]);
        pix_meas.shape = (1,2);
        pt_det_zone_pix[idx, :] = pix_meas

    return pt_det_zone_pix


# Compute overlapping score:
# Actually it is the Union \ Intersection in order to try to match similar object together
# If 0: Then object overlap perfectly
# If >0: Overlap is not perfect
def overlap_mask(mask_1, mask_2):

    # Overlap Mask
    overlap_mask = np.logical_and(mask_1 == 1, mask_2 == 1);

    # Mask of region: Mask_i \ Overlap
    mask_count_1 = (np.logical_and(mask_1 == 1, overlap_mask == 0));
    mask_count_2 = (np.logical_and(mask_2 == 1, overlap_mask == 0));

    # Count the 1
    count_1 =  np.count_nonzero(mask_count_1);
    count_2 =  np.count_nonzero(mask_count_2);

    # Weight more the regions of mask_1 going out of the region of mask_2 (e.g we want mask_1 to be inside mask_2)
    count =  4*count_1 + count_2;

    return count, mask_count_1, mask_count_2;

# Compute the overlap cost for mono image between
#     - mask from object detection (mask RCNN)
#     - mask of the 3D box projected on the image
def compute_cost_mono(opti_params, im_size, cam_model, mask, param_fix):

    # Get optim paramters
    phi = opti_params[0];
    x = opti_params[1];
    y = opti_params[2];

    # Get fixed paramters
    z = param_fix[0];
    l = param_fix[1];
    w = param_fix[2];
    h = param_fix[3];

    # Create Param 3D box
    p_3Dbox = [phi, x, y, z, l, w, h];

    # Create 3D box corners points
    list_pt_F = create_3Dbox(p_3Dbox);

    # Mask box
    mask_3Dbox = create_mask(im_size, cam_model, list_pt_F);
    overlap_score, masko_1, masko_2 = overlap_mask(mask, mask_3Dbox);

    # Return overlap_score
    return overlap_score;


# Compute overlapping percentage between two masks:
# Actually it is the Intersection divided by Union
def overlap_percentage_mask(mask_1, mask_2):

    # Overlap Mask
    overlap_mask = np.logical_and(mask_1 == 1, mask_2 == 1);

    # Union mask
    union_mask = np.logical_or(mask_1 == 1, mask_2 == 1);

    # Count the 1
    count_overlap =  np.count_nonzero(overlap_mask);
    count_union =  np.count_nonzero(union_mask);

    # Weight more the regions of mask_1 going out of the region of mask_2 (e.g we want mask_1 to be inside mask_2)
    percent_overlap =  float(count_overlap) / float(count_union);

    return percent_overlap;

# Compute the overlap percentage for mono image between
#     - mask from object detection (mask RCNN)
#     - mask of the 3D box projected on the image
def compute_percent_overlap(p_3Dbox, im_size, cam_model, mask):

    # Create 3D box corners points
    list_pt_F = create_3Dbox(p_3Dbox);

    # Mask box
    mask_3Dbox = create_mask(im_size, cam_model, list_pt_F);
    percent_overlap = overlap_percentage_mask(mask, mask_3Dbox);

    # Return percent_overlap
    return percent_overlap;


# Compute the overlap cost for steareo images between
#     - mask from object detection (mask RCNN)
#     - mask of the 3D box projected on the image
# Cost is the sum of the cost on each image (but one position / orientation in 3D of the 3DBox)
def compute_cost_stero(opti_params, im_size_1, im_size_2, cam_model_1, cam_model_2, mask_1, mask_2,param_fix):

    # Compute cost for each image:
    cost_1 = compute_cost_mono(opti_params, im_size_1, cam_model_1, mask_1, param_fix);
    cost_2 = compute_cost_mono(opti_params, im_size_2, cam_model_2, mask_2, param_fix)

    # Sum the cost:
    total_cost = cost_1 + cost_2;

    # Return total_cost
    return total_cost;


# Find 3D box correspondig to a mask
# Return the param_3Dbox: [phi, x, y, z, l, w, h]
#    phi: Orientation of the box (yaw) - degrees
#    x  : Position X in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
#    y  : Position Y in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
#    z  : Position Z in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
#    l  : Length of the 3D Box - meters
#    w  : Width of the 3D Box - meters
#    h  : Height of the 3D Box - meters
def find_3Dbox(mask, roi, cam_model, im_size, box_size_lwh):

    # mask: bool array of the size of the image
    # roi: corner coordinates of the ROI (2D bounding box) of the detected object
    # cam_model: camera model
    # im_size: Image size
    # box_size_lwh: 3D box size [length, width, height]

    # Get ROI coordinates
    x_1 = int(roi[1]);
    y_1 = int(roi[0]);
    x_2 = int(roi[3]);
    y_2 = int(roi[2]);

    tl = (x_1, y_1)
    br = (x_2, y_2)

    # Compute rough 3D position of the object:
    # Re-project the center of the ROI on the ground

    # Get center of ROI
    pt_image_x = (x_1 + x_2)/2
    pt_image_y = (y_1 + y_2)/2

    # 3D position by re-projection on the ground
    pt_image = (int(pt_image_x), int(pt_image_y));
    pos_FNED_init = cam_model.projection_ground(0, pt_image);
    pos_FNED_init.shape = (1,3);

    # Construct initial 3D box:
    phi = float(10.0);               # Orientation of the box (yaw) - degrees
    x = float(pos_FNED_init[0,0]);   # Position X in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
    y = float(pos_FNED_init[0,1]);   # Position Y in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
    z = float(0.0);                  # Position Z in F_NED of the center of the bottom side (defined implicitly by the camera paramters) - meters
    l = float(box_size_lwh[0]);          # Length of the 3D Box - meters
    w = float(box_size_lwh[1]);          # Width of the 3D Box - meters
    h = float(box_size_lwh[2]);          # Height of the 3D Box - meters

    # Run the optimization with different Initial Guess
    # Allow to avoid getting stuck on local optimum
    # Especially on the orientation (yaw), as the optimizer has a hard time find good orientation
    param_min = None;
    for phi in range(0,180,60):

        # We only optimize Yaw, Position X, Position Y
        param_opt = [phi, x, y];

        # Position Z assume to be 0
        # 3DBox size: Defined by the class ID
        param_fix = [z, l, w, h];
        p_init = param_opt;

        # Run optimizer: Good method: COBYLA, Powell
        param = opt.minimize(compute_cost_mono, p_init, method='Powell', args=(im_size, cam_model, mask, param_fix), options={'maxfev': 1000, 'disp': True});
        if param_min is None:
            param_min = param;

        # Keep the best values among the different run with different initial guesses
        if param.fun < param_min.fun:
            param_min = param;

    param = param_min;

    # Retrieve 3D box parameters:
    phi = round(param.x[0],4);
    x = round(param.x[1],4);
    y = round(param.x[2],4);

    z = round(param_fix[0],4);
    l = round(param_fix[1],4);
    w = round(param_fix[2],4);
    h = round(param_fix[3],4);

    # Construct param_3Dbox
    param_3Dbox = [phi, x, y, z, l, w, h];

    return param_3Dbox;

# Scale the Mask according to the scaling factor used for the image
def scale_mask(scale_x, scale_y, mask):

    mask = mask.astype('uint8');
    mask = cv2.resize(mask,None,fx=scale_x, fy=scale_y, interpolation = cv2.INTER_CUBIC)
    mask = mask.astype(np.bool);

    return mask;

# Scale the ROI according to the scaling factor used for the image
# [y1, x1, y2, x2]
def scale_roi(scale_x, scale_y, roi):

    roi[0]  = roi[0]*scale_y;
    roi[1]  = roi[1]*scale_x;
    roi[2]  = roi[2]*scale_y;
    roi[3]  = roi[3]*scale_x;

    return roi


# Find 3D box correspondig to a mask
# Adapted for a mutlithread pool input format (one dict)
def find_3Dbox_multithread(input_dict):

    # Extract objetc from input_dict:
    mask = input_dict['mask'];
    roi =  input_dict['roi'];
    cam_model = input_dict['cam_model'];
    im_size = input_dict['im_size'];
    box_size_lwh = input_dict['box_size'];

    # Compute the 3D box:
    param_3Dbox = find_3Dbox(mask, roi, cam_model, im_size, box_size_lwh);
    percent_overlap = compute_percent_overlap(param_3Dbox, im_size, cam_model, mask);

    # Construct result dict:
    result = {};
    result['box_3D'] = param_3Dbox;
    result['mask'] = mask;
    result['det_id'] = input_dict['det_id'];
    result['percent_overlap'] = percent_overlap;

    # Return result
    return result;
