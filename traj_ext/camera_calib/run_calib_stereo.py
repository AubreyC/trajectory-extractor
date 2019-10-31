# import the necessary packages
import argparse
import cv2
import sys
import os
import numpy as np
import time
import copy
from scipy.optimize import linear_sum_assignment

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../'));

from traj_ext.utils import mathutil
from traj_ext.camera_calib.calib_utils import *
from traj_ext.tracker import cameramodel as cm

IMG_1_PATH = os.path.join(ROOT_DIR,'camera_calib/calib_input/varna_area1_camera_sat.png');
CAMERA_CFG_1_PATH = os.path.join(ROOT_DIR,'camera_calib/calib_file/varna_sat_cfg.yml');

IMG_2_PATH = os.path.join(ROOT_DIR,'camera_calib/calib_input/varna_area1_camera_street.jpg');

OUTPUT_CFG_PATH = os.path.join(ROOT_DIR,'camera_calib/calib_file/varna_area1_camera_street_cfg.cfg');
OUTPUT_IMG_PATH = os.path.join(ROOT_DIR,'camera_calib/calib_input/varna_area1_camera_street_calib.png');

def click(event, x, y, flags, param):

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed

    if event == cv2.EVENT_LBUTTONDOWN:
        pt_image = (x, y)
        pt_image_list.append(pt_image);

        # draw a rectangle around the region of interest
        cv2.putText(image, str(len(pt_image_list)), pt_image, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.circle(image, pt_image, 2, (0,0, 255), -1)
        cv2.imshow("image", image)


        # pos_FNED = cam_model_1.projection_ground(0, pt_image);
        # pos_FNED.shape = (1,3);
        # model_points_FNED = np.append(model_points_FNED, pos_FNED, axis=0);
        # print model_points_FNED
        # print pt_image_list


def click_2(event, x, y, flags, param):

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed

    if event == cv2.EVENT_LBUTTONDOWN:
        pt_image_2 = (x, y);
        pt_image_2_list.append(pt_image_2);

        # draw a rectangle around the region of interest
        cv2.putText(image_2, str(len(pt_image_2_list)), pt_image_2, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        cv2.circle(image_2, pt_image_2, 2, (0,0, 255), -1)
        cv2.imshow("image_2", image_2)


if __name__ == '__main__':


   # Print instructions
    print("############################################################\n")
    print("Camera Calibration Interact Software:")
    print("calibrate one camera relatively from an other camera\n")

    print("\nInstruction:")
    print("    - Clik to define key points on Image 1, press Enter when done (min 4 points)")
    print("    - Clik to define the same key points on Image 2, press Enter when done (min 4 points)")
    print("    - Press Enter to save the calibration, or q to exit without saving\n")


    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default= IMG_1_PATH, help="Path to the image")
    parser.add_argument('--camera_1_cfg',dest="camera_1_cfg", default=CAMERA_CFG_1_PATH, type=str, help='Camera 1 yml config file');
    args = parser.parse_args();

    # Intrinsic camera parameters
    fs_read = cv2.FileStorage(args.camera_1_cfg, cv2.FILE_STORAGE_READ)
    cam_matrix_1 = fs_read.getNode('camera_matrix').mat()
    rot_CF1_F = fs_read.getNode('rot_CF_F').mat()
    trans_CF1_F = fs_read.getNode('trans_CF_F').mat()
    dist_coeffs_1 = fs_read.getNode('dist_coeffs').mat()
    # Construct camera model
    cam_model_1 = cm.CameraModel(rot_CF1_F, trans_CF1_F, cam_matrix_1, dist_coeffs_1);


    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(args.image)
    cv2.namedWindow("image")


    pt_image_list = [];

    cv2.setMouseCallback("image", click)

    # keep looping until the 'q' key is pressed
    print('\n')
    print("    - Clik to define key points on Image 1, press Enter when done (min 4 points)")
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'Enter' key is pressed, break
        if key == 13:
            break

        elif  key == ord("q"):
            sys.exit();


    model_points_FNED = np.array([]);
    model_points_FNED.shape = (0,3);
    for pt_image in pt_image_list:

        pos_FNED = cam_model_1.projection_ground(0, pt_image);
        pos_FNED.shape = (1,3);
        model_points_FNED = np.append(model_points_FNED, pos_FNED, axis=0);


    # close all open windows
    #cv2.destroyAllWindows()

    # load the image, clone it, and setup the mouse callback function
    image_2 = cv2.imread(IMG_2_PATH)
    cv2.namedWindow("image_2")

    pt_image_2_list = [];
    cv2.setMouseCallback("image_2", click_2)

    # keep looping until the 'q' key is pressed
    print('\n')
    print("    - Clik to define the same key points on Image 2, press Enter when done (min 4 points)")
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image_2", image_2)
        key = cv2.waitKey(1) & 0xFF
        # if the 'Enter' key is pressed, break
        if key == 13:
            break

        elif  key == ord("q"):
            sys.exit();

    image_points = np.array([]);
    image_points.shape = (0,2);
    for pt_image_2 in pt_image_2_list:

        daz = np.array([pt_image_2[0], pt_image_2[1]]);
        daz.shape = (1,2);
        image_points = np.append(image_points, daz, axis=0);

    im = copy.copy(image_2);
    model_points_F = model_points_FNED;


    print('model_points_F')
    print(model_points_F)
    print('image_points')
    print(image_points)

    ###################################################
    #
    # Calibration
    #
    ###################################################


    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    im_size = im.shape; # Getting Image size
    rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs, image_points_reproj = find_camera_params_opt(im_size, image_points, model_points_F);

    # Convert rotation vector in rotation matrix
    rot_CF_F = cv2.Rodrigues(rot_vec_CF_F)[0];
    print('rot_CF_F:')
    print(rot_CF_F)

    # Convert rotation matrix in euler angle:
    euler_CF_F = mathutil.rotationMatrixToEulerAngles(rot_CF_F);
    print('euler_CF_F: ')
    print(euler_CF_F)

    # Position of the origin expresssed in CF
    print('trans_CF_F (position of the origin expressed in CF): ')
    print(trans_CF_F)

    # Show the origin axis on image
    im = display_NED_frame(im, rot_vec_CF_F, trans_CF_F, camera_matrix, dist_coeffs);

    #Show the keypoints
    im = display_keypoints(im, image_points_reproj, image_points);


    cv2.imshow("Output", im)

    print("    - Press Enter to save the calibration, or q to exit without saving\n")
    key = cv2.waitKey(0);
    save_bool = False;
    # if the 'Enter' key is pressed, end of the program
    if key == 13:
        save_bool = True;

    if save_bool:
        # Define output name
        #output_path = image_path.split('.')[0] + '_cfg.yml';
        output_path = OUTPUT_CFG_PATH;

        # Save the parameters
        save_camera_calibration(output_path, rot_CF_F, trans_CF_F, camera_matrix, dist_coeffs);

        # Save image with key-points
        # im_calib_path = image_path.split('.')[0] + '_calib.' + image_path.split('.')[-1];
        im_calib_path = OUTPUT_IMG_PATH;
        cv2.imwrite(im_calib_path, im);
        print('\n Image config file saved %s \n' %(im_calib_path))


    cv2.destroyAllWindows()
    print("Program Exit\n")
    print("############################################################\n")