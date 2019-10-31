# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-21 18:23:24
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-15 22:13:55

from traj_ext.box3D_fitting import box3D_object
from traj_ext.tracker import cameramodel

import cv2
import os

# Define output dir for the test
DIR_PATH = os.path.dirname(__file__);
OUTPUT_DIR_PATH = os.path.join(DIR_PATH,'test_output');

def test_box3dD(show_image = False):

    # Path to detection and image:
    cam_model_path = 'traj_ext/camera_calib/calib_file/brest/brest_area1_street_cfg.yml';
    image_path = 'traj_ext/camera_calib/calib_file/brest/brest_area1_street.jpg';

    # Open the detection:
    box3D = box3D_object.Box3DObject(0.3,2.3,-4.3,0.0,4.0,2.0,1.0);
    box3D_list = [box3D];

    # Open image and camera model
    cam_model = cameramodel.CameraModel.read_from_yml(cam_model_path);
    image = cv2.imread(image_path);

    # Display on image
    box3D.display_on_image(image, cam_model);

    if show_image:
        cv2.imshow('Test 3D box', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass;

def test_box3D_csv():

    box3D_list = [box3D_object.Box3DObject(0.1,2.3,6.3,0.0,2.0,3.0,1.0), \
                  box3D_object.Box3DObject(0.32,2.3,7.3,0.0,2.0,3.0,1.0),\
                  box3D_object.Box3DObject(-1.3,5.3,2.3,0.0,2.0,3.0,1.0)];

    # Create outut dir if not created
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

    # Save test copy
    csv_path = os.path.join(OUTPUT_DIR_PATH, 'test_box3D.csv');
    box3D_object.Box3DObject.to_csv(csv_path, box3D_list);


if __name__ == '__main__':

    test_box3dD(True);
    test_box3D_csv();