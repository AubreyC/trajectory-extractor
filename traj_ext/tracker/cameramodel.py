# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-02-02 15:23:04
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

# Define a pin-hole camera model
# Utility functions such as projection, etc
# Ref: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

import numpy as np
import math
import cv2
import sys
import os

from traj_ext.utils.mathutil import *

###################################################################
# Camera Util function
###################################################################
# Ref: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

# Pin-hole camera model:
# s*[p_x; p_y; 1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];
# s*[p_x; p_y; 1] = cam_matrix*rot_CF_F*[pos_F + (rot_CF_F^-1)*trans_CF_F];

# Camera Frame: Frame attached to the center of the camera
# F frame: Arbitrary cartesian frame (usually a NED frame)


# Compute the re-pojection of a pixel point to the 3D frame F, according to a constraint on the Z coordinates of the 3D point in F.
def projection_ground(cam_matrix, rot_CF_F, trans_CF_F, z_F, pixel_xy):
    # F : Frame in which the vehicles are moving, attached to the ground
    # CF: Camera Frame

    # cam_matrix: 3x3 camera matrix
    # cam_dist: distortion coefficients
    # rot_CF_F: Rotation Matrix from F to CF
    # trans_CF_F: Position of the origin of F expressed in CF
    # z_F: Constraint on the Z coordinate of the 3D point in F (this enables to re-project from pixel-image plane to 3D position in F)

    # reshape
    pixels = np.append(pixel_xy,1);
    pixels.shape = (3,1);
    trans_CF_F.shape = (3,1);

    # Pin-hole camera model:
    # s*[p_x; p_y; 1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];
    # s*[p_x; p_y; 1] = cam_matrix*rot_CF_F*[pos_F + (rot_CF_F^-1)*trans_CF_F];

    # Let's define F2:
    # Origin: Center of the camera (same as Camera Frame)
    # Axis: Same as F
    # pos_F2 = pos_F + (rot_CF_F^-1)*trans_CF_F; with rot_CF_F = rot_CF_F2

    # Theory:
    # Pin-hole camera: s*[p_x; p_y; 1] = cam_matrix*[rot_CF_F2]*[pos_F2];
    # <=> [p_x; p_y; 1] = cam_matrix*[rot_CF_F2]*[pos_F2/s];
    # <=> rot_CF_F2^-1*cam_matrix^-1*[p_x; p_y; 1] = [pos_F2/s];

    # Actually not really in F, but more in a frame with same axis as F but same origin as CF
    pos_F2_over_s = np.linalg.inv(rot_CF_F).dot(np.linalg.inv(cam_matrix).dot(pixels));

    # Compute z_F2 from Z_f constraint and caemra position
    # pos_F2 = pos_F + (rot_CF_F^-1)*trans_CF_F
    z_F2 = z_F + rot_CF_F.transpose().dot(trans_CF_F)[2,0];

    # We have: z_F2 and z_F2/s:
    # Get the value of s
    s = z_F2/pos_F2_over_s[2,0];

    # Now we have s and pos_F2/s, we can copute pos_F2 bu multiplying by s:
    # Compute full position in F2
    pos_F2 = s*pos_F2_over_s;

    # Convret position in F2 in F:
    pos_F = pos_F2 - rot_CF_F.transpose().dot(trans_CF_F);

    return pos_F

# Compute the pojection of a 3D point expressed in frame F on the image plane (pixel point)
# Pin-hole camera model:
# s*[p_x; p_y; 1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];
def projection(cam_matrix, rot_CF_F, trans_CF_F, pos_F):
    # F : Frame in which the vehicles are moving, attached to the ground
    # CF: Camera Frame

    # cam_matrix: 3x3 camera matrix
    # cam_dist: distortion coefficients
    # rot_CF_F: Rotation Matrix from F to CF
    # trans_CF_F: Position of the origin of CF expressed in F

    # Eg: [p_x; p_y; 1] = cam_matrix*rot_CF_F*(1/s)*[pos_F + trans_CF_F]
    pos_F.shape = (3,1);
    trans_CF_F.shape = (3,1);

    # Pin-hole camera model:
    # [s*p_x; s*p_y; s*1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];
    pixel_s = projection_with_s(cam_matrix, rot_CF_F, trans_CF_F, pos_F);

    # s is the thrid component of [s*p_x; s*p_y; s*1]
    # So we can just divide the first component by "s" to get the pixel coordinates
    pixel = pixel_s/float(pixel_s[2]);

    pixel =  np.asarray(pixel[0:2,0]).reshape(-1);

    # Cast into integer but carefull to round before
    pixel = np.around(pixel).astype(dtype =int)

    return pixel

# Compute the pojection with the sacling factor s of a point expressed in frame F on the image plane
# Pin-hole camera model:
# [s*p_x; s*p_y; s*1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];
def projection_with_s(cam_matrix, rot_CF_F, trans_CF_F, pos_F):
    # F : Frame in which the vehicles are moving, attached to the ground
    # CF: Camera Frame

    # cam_matrix: 3x3 camera matrix
    # cam_dist: distortion coefficients
    # rot_CF_F: Rotation Matrix from F to CF
    # trans_CF_F: Position of the origin of CF expressed in F

    # Eg: [s*p_x; s*p_y; s] = cam_matrix*rot_CF_F*[pos_F + trans_CF_F]
    pos_F.shape = (3,1);
    trans_CF_F.shape = (3,1);

    # Pin-hole camera model:
    # [s*p_x; s*p_y; s*1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];
    pixel_s = cam_matrix.dot(rot_CF_F.dot(pos_F) + trans_CF_F);

    return pixel_s

# Compute the sacling factor "s" of the projection of a 3D point expressed in frame F on the image plane (pixels coordinates)
# Pin-hole camera model:
# [s*p_x; s*p_y; s*1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];
def compute_s(cam_matrix, rot_CF_F, trans_CF_F, pos_F):
    # F : Frame in which the vehicles are moving, attached to the ground
    # CF: Camera Frame

    # cam_matrix: 3x3 camera matrix
    # cam_dist: distortion coefficients
    # rot_CF_F: Rotation Matrix from F to CF
    # trans_CF_F: Position of the origin of CF expressed in F
    # pos_F: 3D position expressed in frame F

    # Reshape
    pos_F.shape = (3,1);
    trans_CF_F.shape = (3,1);

    # [s*p_x; s*p_y; s*1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];
    pixel_s =  pixel_s = projection_with_s(cam_matrix, rot_CF_F, trans_CF_F, pos_F);
    return pixel_s[2,0];

# In 2D, display the ellispe corresponding to the covariance matrix
def compute_ellipse(img, chisquare_val, pt_xy, mat_cov):
        #From: https://gist.github.com/eroniki/2cff517bdd3f9d8051b5
        # Inputs:
        # img: img to draw the ellispe on
        # chisquare_val: confidence indice since e normalized follow a Chi-Sqaured distribution
        # pt_xy: center of the ellispe
        # mat_cov: Covaraince matrix

    # Compute eigenvalues and eigenvectors
    e_val, e_vec = np.linalg.eig(mat_cov);

    # Reorder to get max eigen first
    if e_val[1] > e_val[1]:
        e_val = reverse(e_val);
        e_vec = reverse(e_vec);

    # Compute angle between eigenvector and x axis
    angle = -np.arctan2(e_vec[0,1],e_vec[0,0]);

    # Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
    if angle < 0 :
        angle += 6.28318530718;

    halfmajoraxissize = int(chisquare_val*np.sqrt(math.fabs(e_val[0])));
    halfminoraxissize = int(chisquare_val*np.sqrt(math.fabs(e_val[1])));

    #print (halfmajoraxissize, halfminoraxissize)
    #cv2.ellipse(img,pt_xy,(halfmajoraxissize, halfminoraxissize), angle,0,360,255, 2)
    cv2.ellipse(img,pt_xy,(halfmajoraxissize, halfminoraxissize), np.rad2deg(angle),0,360,255, 2)

# Coordiante frame conversion: from Camera Frame (CF) to frame F
def convert_CF_to_F(rot_CF_F, trans_CF_F, pos_CF_31):

    # rot_CF_F: Rotation Matrix from F to CF
    # trans_CF_F: Position of the origin of CF expressed in F
    # pos_CF_31: 3D position expressed in frame CF

    # Reshape
    pos_CF_31.shape = (3,1);
    trans_CF_F.shape = (3,1);
    rot_CF_F.shape = (3,3);

    # Change of frame
    #x_CF = rot_CF_F*(X_F - trans_CF_F);
    pos_F_31 = rot_CF_F.transpose().dot((pos_CF_31 - trans_CF_F))

    return pos_F_31;

# Scale the camera matrix for a re-sized image
def scale_camera_matrix(scale_x, scale_y, cam_matrix):

    cam_matrix[0,0] = scale_x*cam_matrix[0,0];
    cam_matrix[0,2] = scale_x*cam_matrix[0,2];
    cam_matrix[1,1] = scale_y*cam_matrix[1,1];
    cam_matrix[1,2] = scale_y*cam_matrix[1,2];

    return cam_matrix;

def display_NED_frame(image, rot_CF_F, trans_CF_F, camera_matrix, dist_coeffs, length = 10.0):
    """Display NED on the image according to camera model parameters

    Args:
        image (TYPE): Image form camera
        rot_CF_F (TYPE): Rotation matrix
        trans_CF_F (TYPE): Translation matrix
        camera_matrix (TYPE): Camera matrix (intrinsic parameters)
        dist_coeffs (TYPE): Distorsion coefficients
        length (float, optional): Lenght of the axis - Default: 1.0 m

    Returns:
        TYPE: Image
    """

    # Origin axis on image
    (pt_origin, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rot_CF_F, trans_CF_F, camera_matrix, dist_coeffs)
    (pt_test_2, jacobian) = cv2.projectPoints(np.array([(length, 0.0, 0.0)]), rot_CF_F, trans_CF_F, camera_matrix, dist_coeffs)
    (pt_test_3, jacobian) = cv2.projectPoints(np.array([(0.0, length, 0.0)]), rot_CF_F, trans_CF_F, camera_matrix, dist_coeffs)

    pt_o = ( int(pt_origin[0][0][0]), int(pt_origin[0][0][1]))
    pt_x = ( int(pt_test_2[0][0][0]), int(pt_test_2[0][0][1]))
    pt_y = ( int(pt_test_3[0][0][0]), int(pt_test_3[0][0][1]))

    # Add line of axis
    cv2.line(image, pt_o, pt_x, (255,0,0), 2)
    cv2.line(image, pt_o, pt_y, (255,255,0), 2)

    return image;

###################################################################
# Camera Model class
###################################################################

# Define a camera model with:

# cam_matrix: 3x3 camera matrix
# cam_dist: distortion coefficients
# rot_CF_F: Rotation Matrix from F to CF
# trans_CF_F: Position of the origin of CF expressed in F
# dist_coeffs: Distorsion coefficients (assume to be 0 here)

# Ref: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
class CameraModel:

    #Init with camera parameters
    def __init__(self, rot_CF_F, trans_CF_F, cam_matrix, dist_coeffs):

        # Rotation: From frame to Camera Frame
        if rot_CF_F.shape != (3,3):
            raise ValueError('Camera Model: rot_CF_F size error', rot_CF_F.shape)
        self.rot_CF_F = rot_CF_F;
        self.rot_vec_CF_F = cv2.Rodrigues(rot_CF_F)[0];


        #Translation: Origin of F expressed in CF
        if trans_CF_F.shape != (3,1):
            raise ValueError('Camera Model: trans_CF_F size error', trans_CF_F.shape)
        self.trans_CF_F = trans_CF_F;

        # Camera Matrix
        if cam_matrix.shape != (3,3):
            raise ValueError('Camera Model: cam_matrix size error', cam_matrix.shape)
        self.cam_matrix = cam_matrix;

        if dist_coeffs.shape != (4,1):
            raise ValueError('Camera Model: dist_coeffs size error', dist_coeffs.shape);
        self.dist_coeffs = dist_coeffs; # Not use for now, assumed to be 0

    @classmethod
    def read_from_yml(cls, input_path):
        """Read Camera model from yml config file

        Args:
            input_path (TYPE): Description

        Returns:
            TYPE: Camera Model

        Raises:
            ValueError: Wrong path
        """

        # Check input path
        if input_path == '':
            raise ValueError('[Error] Camera input path empty: {}'.format(input_path));

        # Intrinsic camera parameters
        fs_read = cv2.FileStorage(input_path, cv2.FILE_STORAGE_READ)
        cam_matrix = fs_read.getNode('camera_matrix').mat()
        rot_CF_F = fs_read.getNode('rot_CF_F').mat()
        trans_CF_F = fs_read.getNode('trans_CF_F').mat()
        dist_coeffs = fs_read.getNode('dist_coeffs').mat()

        # Some checks:
        if cam_matrix is None:
            raise ValueError('[Error] Camera cfg input path is wrong: {}'.format(input_path));

        # Construct camera model
        cam_model = CameraModel(rot_CF_F, trans_CF_F, cam_matrix, dist_coeffs);

        return cam_model;

    def save_to_yml(self, output_path):

        # Write Camera Param in YAML file
        fs_write = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
        fs_write.write('rot_CF_F', self.rot_CF_F);
        fs_write.write('trans_CF_F', self.trans_CF_F);
        fs_write.write('camera_matrix', self.cam_matrix);
        fs_write.write('dist_coeffs', self.dist_coeffs);
        fs_write.release()
        print('\n Camera config file saved %s \n' %(output_path))

    def display_NED_frame(self, image):
        """Display the NED frame on the image according to the camera model

        Args:
            image (TYPE): Image from the camera

        Returns:
            TYPE: Image with NED frame on it
        """
        return display_NED_frame(image, self.rot_CF_F, self.trans_CF_F, self.cam_matrix, self.dist_coeffs);

    # Apply scaling factor to the camera matrix, used when changin image size
    def apply_scale_factor(self, scale_x, scale_y):

        if scale_x < 0 or scale_y < 0:
            print('[ERROR]: apply_scale_factor must be > 0: scale_x: {} scale_y: {}'.format(scale_x, scale_y));
            return;

        self.cam_matrix = scale_camera_matrix(scale_x, scale_y, self.cam_matrix);

    def project_list_pt_F(self, list_pt_F):

        pt_img_list = [];
        for pt in list_pt_F:
                # Project 3Dbox corners on Image Plane
                pt.shape = (1,3);
                (pt_img, jacobian) = cv2.projectPoints(pt, self.rot_CF_F, self.trans_CF_F, self.cam_matrix, self.dist_coeffs)

                # Cretae list of tulpe
                pt_img_tulpe = (int(pt_img[0][0][0]), int(pt_img[0][0][1]));
                pt_img_list.append(pt_img_tulpe);


        return pt_img_list;

    # Compute the pojection of a 3D point expressed in frame F on the image plane (pixel point)
    def project_points(self, pos_31):

        pos_31.shape = (3,1);

        # Pin-hole camera model:
        # s*[p_x; p_y; 1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];

        # Use openCV function, but this is equivalent to the projection in CameraModel
        # (pt_current, jacobian) = cv2.projectPoints(pos_31, self.rot_vec_CF_F, self.trans_CF_F, self.cam_matrix, self.dist_coeffs)

        # See custom projection function defined above
        pt_current = projection(self.cam_matrix, self.rot_CF_F, self.trans_CF_F, pos_31);
        return pt_current;

    # Compute the re-pojection of a 2D pixel point to the 3D frame F, according to a constraint on the Z coordinates of the 3D point in F.
    def projection_ground(self, z_F, pixel_xy):

        # Call project_ground from camera model:
        pos_F = projection_ground(self.cam_matrix, self.rot_CF_F, self.trans_CF_F, z_F, pixel_xy);
        return pos_F;

    # Compute the re-pojection of a 2D pixel point to the 3D frame CF, according to a constraint on the Z coordinates of the 3D point in CF.
    def projection_3D_CF(self, z_CF, pixel_xy):

        # Call project_3D_CF from camera model:
        rot_CF_F = np.identity(3);
        trans_CF_F = np.zeros((3,1));

        # Call projection_ground with a rot_CF_F=I and trans_CF_F = 0 such that frame F is the camera frame (CF)
        pos_CF = projection_ground(self.cam_matrix, rot_CF_F, trans_CF_F, z_CF, pixel_xy);
        return pos_CF;

    # Coordiante frame conversion: from Camera Frame (CF) to frame F
    def convert_CF_to_F(self, pos_CF_31):

        pos_CF_31.shape = (3,1);

        # Use openCV function, but this is equivalent to the projection in CameraModel
        # (pt_current, jacobian) = cv2.projectPoints(pos_31, self.rot_vec_CF_F, self.trans_CF_F, self.cam_matrix, self.dist_coeffs)
        pos_F_31 = convert_CF_to_F(self.rot_CF_F, self.trans_CF_F, pos_CF_31);
        return pos_F_31;


    # Compute the sacling factor "s" of the projection of a 3D point expressed in frame F on the image plane (pixels coordinates)
    # Pin-hole camera model:
    # [s*p_x; s*p_y; s*1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];
    def compute_s(self, pos_F):
        return compute_s(self.cam_matrix, self.rot_CF_F, self.trans_CF_F, pos_F);


    # EKF Helper
    # Projection function: pix = h(pos_F)
    # With pix = [pix_u, pix_v];
    #      pos_F = [pos_x, pos_y, pos_z];
    # Compute the derivative: d(pix)/d(pos_F) at pos_F
    # H = [d(pix_u)/d(pos_x) d(pix_u)/d(pos_y)  d(pix_u)/d(pos_z);
    #     [d(pix_v)/d(pos_x) d(pix_v)/d(pos_y)  d(pix_v)/d(pos_z)];
    def compute_meas_H(self, pos_F):

        # Theory: Use multiplacation rule of derivation
        # d(s*pix)/d(pos_F)= d(s)/d(pos_F)*pix + s*d(pix)/d(pos_F)
        # <=> d(pix)/d(pos_F) = (1/s)*(d(s*pix)/d(pos_F) - d(s)/d(pos_F)*pix)

        # And:
        # [s*p_u; s*p_v; s*1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];
        # [s*p_u; s*p_v; s*1] = cam_matrix*[rot_CF_F]*[pos_F] + trans_CF_F;
        # So:
        # d(s*pix)/d(pos_F) = cam_matrix*rot_CF_F
        d_pix_with_s = self.cam_matrix.dot(self.rot_CF_F);

        # Comtute s:
        pos_F.shape = (3,1);
        s = self.compute_s(pos_F);

        # d(s*pix)/d(pos_F) =  [d(s*p_u); d(s*p_v); d(s)]/d(pos_F)
        # ds is actually the last row of d_pix_with_s
        ds = d_pix_with_s[2,:];
        ds.shape = (1,3)

        # Compute pix:
        pix = self.project_points(pos_F);
        pix_31 = np.append(pix, 1);
        pix_31.shape = (3,1);

        #  d(pix)/d(pos_F) = (1/s)*(d(s*pix)/d(pos_F) - d(s)/d(pos_F)*pix)
        d_pix = (1/s)*(d_pix_with_s - pix_31.dot(ds))

        # Only keep the two first rows
        # d(pix)/d(pos_F) =  [d(p_x); d(p_y); d(1)]/d(pos_F)
        # We want d([p_u;p_v])/d([pos_x,pos_y,pos_z])
        H = np.zeros((2,3));
        H = d_pix[:2,:];

        return H;

    # def compute_meas_H_bis(self, pos_F):

    #     # Theory: Use multiplacation rule of derivation
    #     # d(s*pix)/d(pos_F)= d(s)/d(pos_F)*pix + s*d(pix)/d(pos_F)
    #     # <=> d(pix)/d(pos_F) = (1/s)*(d(s*pix)/d(pos_F) - d(s)/d(pos_F)*pix)

    #     # And:
    #     # [s*p_u; s*p_v; s*1] = cam_matrix*[rot_CF_F, trans_CF_F]*[pos_F; 1];
    #     # [s*p_u; s*p_v; s*1] = cam_matrix*[rot_CF_F]*[pos_F] + trans_CF_F;
    #     # So:
    #     # d(s*pix)/d(pos_F) = cam_matrix*rot_CF_F

    #     # Comtute s:
    #     pos_F.shape = (3,1);
    #     s = self.compute_s(pos_F);

    #     # Compute pix:
    #     pix = self.project_points(pos_F);
    #     pix_31 = np.append(pix, 1);
    #     pix_31.shape = (3,1);

    #     A = np.array(
    #                   [[1, 0, 0],
    #                    [0, 1, 0]], dtype = "double"
    #                   );
    #     A.shape = (2,3);


    #     B = np.array(
    #                   [0, 0, 1], dtype = "double"
    #                   );
    #     B.shape = (1,3);

    #     #  d(pix)/d(pos_F) = (1/s)*(d(s*pix)/d(pos_F) - d(s)/d(pos_F)*pix)
    #     H = (1/s)*(self.cam_matrix.dot(self.rot_CF_F) - pix_31.dot(B.dot(self.cam_matrix.dot(self.rot_CF_F))));
    #     H = A.dot(H);

    #     return H;
