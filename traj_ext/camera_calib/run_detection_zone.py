# import the necessary packages
import argparse
import cv2
import sys
import os
import numpy as np
import time
import copy
from scipy.optimize import linear_sum_assignment
import configparser

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../'));
sys.path.append(ROOT_DIR);

from camera_calib import calib_utils
from utils import cfgutil
from utils import mathutil
from tracker import cameramodel as cm
from box3D_fitting import Box3D_utils

# Create a default cfg file which holds default values for the path
def create_default_cfg():

    config = configparser.ConfigParser();

    config['INPUT_PATH'] = \
                        {'camera_img_street_path': '',\
                         'camera_cfg_street_path': '', \
                         'camera_img_sat_path': '',\
                         'camera_cfg_sat_path' :'' }

    config['OUTPUT_PATH'] = \
                        {'detection_zone_path': ''};


    # Header of the cfg file
    text = '\
##########################################################################################\n\
#\n\
# DETECTION ZONE CALIBRATION\n\
#\n\
# Please modify this config file according to your configuration.\n\
# Path must be ABSOLUTE PATH\n\
##########################################################################################\n\n'

    # Write the cfg file
    with open('DETECTION_ZONE_CFG.ini', 'w') as configfile:
        configfile.write(text);
        config.write(configfile)
        print('DETECTION_ZONE_CFG.ini created')
        print('Please fill the config file and restart the program.\n')


def click(event, x, y, flags, param ):

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed

    pt_image_list = param[0]
    cam_model_1 = param[1];
    image_1 = param[2];
    cam_model_2 = param[3];
    image_2 = param[4];


    if event == cv2.EVENT_LBUTTONDOWN:
        pt_image = (x, y)
        pt_image_list.append(pt_image);

        draw_image(image_1, image_2, cam_model_1, cam_model_2, pt_image_list);

    return;

def draw_image(image_1, image_2, cam_model_1, cam_model_2, pt_image_list):

    image_1_temp = copy.copy(image_1);
    image_2_temp = copy.copy(image_2);

    pt_FNED_list_temp = [];
    for index, pt_image in enumerate(pt_image_list):
        # for pt_i in pt_image_list:
        pos_FNED = cam_model_1.projection_ground(0, pt_image);
        pos_FNED.shape = (3,1);
        pt_FNED_list_temp.append(pos_FNED);

        # Show mask of the region of interest
        cv2.putText(image_1_temp, str(index), pt_image, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.circle(image_1_temp, pt_image, 2, (0,0, 255), -1)

    if len(pt_FNED_list_temp) > 1:

        mask_1 = Box3D_utils.create_mask((image_1_temp.shape[0], image_1_temp.shape[1]), cam_model_1, pt_FNED_list_temp);
        Box3D_utils.draw_mask(image_1_temp, mask_1, (0,0,255));

        mask_2 = Box3D_utils.create_mask((image_2_temp.shape[0], image_2_temp.shape[1]), cam_model_2, pt_FNED_list_temp);
        Box3D_utils.draw_mask(image_2_temp, mask_2, (0,0,255));


    cv2.imshow("image_2", image_2_temp)
    cv2.imshow("image_1", image_1_temp)


def main():

    # Print instructions
    print("############################################################")
    print("Detection Zone Calibration software")
    print("############################################################\n")

    # ##########################################################
    # # Parse Arguments
    # ##########################################################
    argparser = argparse.ArgumentParser(
        description='Detection Zone Calibrationz')
    argparser.add_argument(
        '-i', '--init',
        action='store_true',
        help='Generate config file')
    argparser.add_argument(
        '-cfg', '--config',
        default='DETECTION_ZONE_CFG.ini',
        help='Path to the config file')
    args = argparser.parse_args();

    if args.init:
        create_default_cfg();
        return;

    # ##########################################################
    # # Read config file:
    # ##########################################################
    config = cfgutil.read_cfg(args.config);

    # Print instructions
    print("Instruction:")
    print("    - Click on the image to define the detection zone")
    print("    - Press 'd' to delete last point")
    print("    - Press Enter to save the detection file")
    print("    - Press 'q' to exit without saving detection zone file \n")

    # Construct camera model
    cam_model_1 = calib_utils.read_camera_calibration(config['INPUT_PATH']['camera_cfg_street_path']);
    cam_model_2 = calib_utils.read_camera_calibration(config['INPUT_PATH']['camera_cfg_sat_path']);

    # load the image, clone it, and setup the mouse callback function
    image_2 = cv2.imread(config['INPUT_PATH']['camera_img_sat_path'])
    if image_2 is None:
        print('\n[Error]: camera_img_sat_path is not valid: {}'.format(config['INPUT_PATH']['camera_img_sat_path']));
        return;

    # load the image, clone it, and setup the mouse callback function
    image_1 = cv2.imread(config['INPUT_PATH']['camera_img_street_path'])
    if image_2 is None:
        print('\n[Error]: camera_img_sat_path is not valid: {}'.format(config['INPUT_PATH']['camera_img_street_path']));
        return;

    # Create windows
    cv2.namedWindow("image_1")
    cv2.namedWindow("image_2")

    pt_image_list = [];
    cv2.setMouseCallback("image_1", click, param=(pt_image_list, cam_model_1, image_1, cam_model_2, image_2))

    cv2.imshow("image_2", image_2)
    cv2.imshow("image_1", image_1)

    # keep looping until the 'q' key is pressed
    save_zone = False;
    while True:

        # display the image and wait for a keypress
        key = cv2.waitKey(1) & 0xFF
        # if key == ord("c"):
        # if the 'Enter' key is pressed, end of the program
        if key == 13:
            save_zone = True;
            break

        elif  key == ord("q"):
            return;

        elif key == ord("d"):
            if len(pt_image_list) > 0:
                pt_image_list.pop();
                draw_image(image_1, image_2, cam_model_1, cam_model_2, pt_image_list);

    # Make sure there is enough point
    if not len(pt_image_list) > 2:
        print('[Error]: Not enough points to define the detection zone (3 points minimum): {}'.format(pt_image_list))
        return;

    model_points_FNED = np.array([]);
    model_points_FNED.shape = (0,3);

    # Convert the point in pixel in point in FNED
    for index, pt_image in enumerate(pt_image_list):
        pos_FNED = cam_model_1.projection_ground(0, pt_image);

        pos_FNED.shape = (1,3);
        model_points_FNED = np.append(model_points_FNED, pos_FNED, axis=0);

    # Only keep convex hull
    model_points_FNED_convex = calib_utils.convex_hull(model_points_FNED);

    if save_zone:
        # Define output name
        output_path = config['OUTPUT_PATH']['detection_zone_path'];
        print("Saving detection zone: {} \n".format(output_path));

        # Write Param in YAML file
        fs_write = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
        fs_write.write('model_points_FNED', model_points_FNED_convex);
        fs_write.release()

    print("Program Exit\n")
    print("############################################################\n")

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

    # except Exception as e:
    #     print('[Error]: {}'.format(e))
