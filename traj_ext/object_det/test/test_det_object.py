# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-21 18:23:24
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-27 22:38:07

from traj_ext.object_det.det_object import DetObject
import cv2
import os

# Define output dir for the test
DIR_PATH = os.path.dirname(__file__);
OUTPUT_DIR_PATH = os.path.join(DIR_PATH,'test_output');

def test_object_det(show_image = False):

    # Path to detection and image:
    det_csv_path = 'test_dataset/brest/brest_20190609_130424_327_334/output/det/csv/brest_20190609_130424_327_334_0000004900_det.csv'
    image_path = 'test_dataset/dataset/brest/brest_20190609_130424_327_334/img/brest_20190609_130424_327_334_0000004900.jpg'

    # Open the detection:
    det_object_list = DetObject.from_csv(det_csv_path);

    image = cv2.imread(image_path);

    for det in det_object_list:
        det.expand_mask();
        det.remove_mask();
        det.display_on_image(image);

    if show_image:
        cv2.imshow('Test Object Det', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass;

    image = cv2.imread(image_path);

    for det in det_object_list:
        det.expand_mask();
        det.display_on_image(image);

    if show_image:
        cv2.imshow('Test Object Det', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass;

def test_object_det_csv():

    # Path to detection and image:
    det_csv_path = 'dataset/brest/brest_20190609_130424_0_900/output/det/csv/brest_20190609_130424_0_900_0000044700_det.csv'
    image_path = 'dataset/brest/brest_20190609_130424_0_900/img/brest_20190609_130424_0_900_0000044700.jpg'

    # Open the detection:
    det_object_list = DetObject.from_csv(det_csv_path);

    for det in det_object_list:
        det.expand_mask();
        det.remove_mask();

    # Create outut dir if not created
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

    # Save test copy
    csv_path = os.path.join(OUTPUT_DIR_PATH, 'test_copy_det.csv');
    DetObject.to_csv(csv_path, det_object_list);


if __name__ == '__main__':

    test_object_det(True);
    test_object_det_csv();