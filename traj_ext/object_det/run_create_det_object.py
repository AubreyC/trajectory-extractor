# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-24 15:34:17
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-16 21:53:11

import copy
import cv2
import argparse
import os
import subprocess
import math;
import numpy as np

from traj_ext.postprocess_track import trajutil

from traj_ext.tracker.cameramodel import CameraModel

from traj_ext.postprocess_track.agent_type_correct import AgentTypeCorrect

from traj_ext.object_det import det_object
from traj_ext.object_det.mask_rcnn import detect_utils

from traj_ext.hd_map.HD_map import HDmap

from traj_ext.postprocess_track.time_ignore import TimeIgnore

from traj_ext.utils import det_zone
from traj_ext.utils import mathutil

def click(event, x, y, flags, param):

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed

    pt_image_list = param[0]
    image = param[1];


    if event == cv2.EVENT_LBUTTONDOWN:
        pt_image = (x, y)
        pt_image_list.append(pt_image);
        # print(pt_image_list)

        draw_image(image, pt_image_list);

    return;

def draw_image(image, pt_image_list):


    if not (image is None):
        image_temp = copy.copy(image);

        for index, pt_image in enumerate(pt_image_list):

            # Show mask of the region of interest
            # cv2.putText(image_temp, str(index), pt_image, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.circle(image_temp, pt_image, 2, (0,0, 255), -1)

        if len(pt_image_list) > 1:

            mask_1 = det_object.create_mask_image((image_temp.shape[0], image_temp.shape[1]), pt_image_list);
            det_object.draw_mask(image_temp, mask_1, (0,0,255));

        cv2.imshow("Det creation", image_temp)

def print_instructions():

    print('\nInstuctions Detection Creation:\
                         \n- Click on image to draw detection mask\
                         \n- d: Delete last point\
                         \n- n: Create detection from current mask\
                         \n- s: Save new detection to detections list\
                         \n- q: Quit without saving new detection\
                         \n')

def create_detection(image, label, frame_name = '', frame_id = None, det_object_list = []):

    print("\n[Detections Creation]")

    det_object_list_new = copy.copy(det_object_list);
    for det in det_object_list_new:
        det.display_on_image(image);

    cv2.imshow("Det creation", image);

    pt_image_list = [];
    cv2.setMouseCallback("Det creation", click, param=(pt_image_list, image))
    print_instructions();

    while True:
        save_flag = False;

        # display the image and wait for a keypress
        key = cv2.waitKey(0) & 0xFF

        # # if the 'Enter' key is pressed, end of the program
        # if key == 13:
        #     break;

        if  key == ord("q"):
            break;

        elif key == ord("d"):
            if len(pt_image_list) > 0:
                pt_image_list.pop();
                draw_image(image, pt_image_list);

        elif key == ord("n"):

            if len(pt_image_list) > 1:
                mask = det_object.create_mask_image((image.shape[0], image.shape[1]), pt_image_list);
                det_id = det_object.DetObject.get_max_det_id(det_object_list_new) + 1;
                det =  det_object.DetObject.from_mask(det_id, label, mask, 0.99, frame_name = frame_name, frame_id = frame_id);

                det_object_list_new.append(det);

                while len(pt_image_list) > 0:
                    pt_image_list.pop();

                for det in det_object_list_new:
                    det.display_on_image(image);

                cv2.imshow("Det creation", image);
            else:
                print("Detections Creation: Not enough points")

        elif key == ord("s"):
            save_flag = True;
            break;

        else:
            print_instructions();

    cv2.destroyWindow('Det creation');
    return det_object_list_new, save_flag;

def main():

    # Arg parser:
    parser = argparse.ArgumentParser(description='Create detections on an image: Used to create area to ignore in the de_association step');
    parser.add_argument('-image', dest="image_path", type=str, help='Path of the image', default = '');
    parser.add_argument('-det_object', dest="det_object_path", type=str, help='Path of existing det_object', default = '');
    parser.add_argument('-output', dest="output_path", type=str, help='Path of the output det_object', default = '');

    args = parser.parse_args()
    if not os.path.isfile(args.image_path):
        print('[ERROR]: Image path {} not correct'.format(args.image_path));
        return False;

    image = cv2.imread(args.image_path);


    det_object_list = [];
    print(args.det_object_path)
    if os.path.isfile(args.det_object_path):
        det_object_list = det_object.DetObject.from_csv(args.det_object_path)

    det_object_list_new, save_flag = create_detection(image, 'manual', det_object_list = det_object_list);

    if save_flag:
        csv_path = args.output_path;
        csv_path = os.path.join(csv_path, 'manual_detection.csv')
        print('Saving detection object: {}'.format(csv_path));
        det_object.DetObject.to_csv(csv_path, det_object_list_new);

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
