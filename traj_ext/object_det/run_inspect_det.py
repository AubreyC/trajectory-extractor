
import copy
import cv2
import argparse
import os
import subprocess
import math;

from traj_ext.postprocess_track import trajutil

from traj_ext.tracker.cameramodel import CameraModel

from traj_ext.postprocess_track.agent_type_correct import AgentTypeCorrect

from traj_ext.object_det.det_object import DetObject
from traj_ext.object_det.mask_rcnn import detect_utils

from traj_ext.visualization.HD_map import HDmap

from traj_ext.postprocess_track.time_ignore import TimeIgnore


from traj_ext.utils import det_zone
from traj_ext.utils import mathutil

def click_detection(event, x, y, flags, param):
    """Click callback to enable / disbale specific detections by clicking on it

    Args:
        event (TYPE): Description
        x (TYPE): Description
        y (TYPE): Description
        flags (TYPE): Description
        param (TYPE): Description

    Returns:
        TYPE: None
    """

    # If clicked
    if event == cv2.EVENT_LBUTTONDOWN:

        # Get parameters
        det_object_list = param[0];
        det_csv_path = param[1];
        image_current = param[2];
        label_list = param[3];
        label_replace = param[4];

        # Copy current image
        image_current_det = copy.copy(image_current);

        # Get current detections
        # det_object_list = DetObject.from_csv(det_csv_path);

        # Enable / Disable detection that corresponds to the click
        for det_object in det_object_list:
            if det_object.is_point_in_det_2Dbox(x, y):

                det_object.good = not det_object.good;
                print('Detection {}: {}'.format(det_object.det_id, det_object.good));

        # Save the detctions to csv
        print('Saving detections: {}'.format(det_csv_path));
        DetObject.to_csv(det_csv_path, det_object_list);

        # Show new detections
        for det in det_object_list:
            if det.label in label_list:
                det.display_on_image(image_current_det);
        cv2.imshow('Detection', image_current_det)

    # If clicked
    if event == cv2.EVENT_RBUTTONDOWN:

        # Get parameters
        det_object_list = param[0];
        det_csv_path = param[1];
        image_current = param[2];
        label_list = param[3];
        label_replace = param[4];

        # Copy current image
        image_current_det = copy.copy(image_current);

        # Get current detections
        # det_object_list = DetObject.from_csv(det_csv_path);

        # Enable / Disable detection that corresponds to the click
        for det_object in det_object_list:
            if det_object.is_point_in_det_2Dbox(x, y):

                det_object.label = label_replace;
                print('Detection {}: {}'.format(det_object.det_id, det_object.label));

        # Save the detctions to csv
        print('Saving detections: {}'.format(det_csv_path));
        DetObject.to_csv(det_csv_path, det_object_list);

        # Show new detections
        for det in det_object_list:
            if det.label in label_list:
                det.display_on_image(image_current_det);
        cv2.imshow('Detection', image_current_det)

    return;

def main():

    # Arg parser:
    parser = argparse.ArgumentParser(description='Visualize the detections');
    parser.add_argument('-image', dest="image_path", type=str, help='Path of the image', default = 'dataset/test/varna_20190125_153327_0_900/img');
    parser.add_argument('-det', dest="det_dir_path", type=str, help='Path to the detection folder', default = 'dataset/test/varna_20190125_153327_0_900/output/test_output/det_crop');
    parser.add_argument('-label', dest="label_list", nargs = '*', type=str, help='List of label to show: boat airplane', default =['traffic light'] );
    parser.add_argument('-label_replace', dest="label_replace", type=str, help='Label used to replace labels', default ='car' );

    args = parser.parse_args()


    print('\nShow detections: {}'.format(args.label_list))
    print('Replace by: {}\n'.format(args.label_replace))

    # Check if det data available:
    det_data_available = os.path.isdir(args.det_dir_path);
    print('Detection data: {}'.format(det_data_available));

    if len(args.label_list) < 1:
        print('[ERROR]: Label list empty: {}'.format(args.label_list))
        return;

    # Check if image is directoty
    image_in_dir = os.path.isdir(args.image_path);

    # If image directory, open list of images
    if image_in_dir:
        list_img_file = os.listdir(args.image_path);
        list_img_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))));

    # Else open the unique image
    else:
        image = cv2.imread(args.image_path);

    frame_index = 0;
    step = 1;
    while True:

        #######################################################
        ## Show Trajectories
        #######################################################

        # Get Time ms
        frame_index = mathutil.clip(frame_index, 0, len(list_img_file)-1);

        if image_in_dir:
            img_file_name = list_img_file[frame_index];
            image_current = cv2.imread(os.path.join(args.image_path, img_file_name));
        else:
            # Copy image
            image_current = image;

        print('Showing: frame_id: {} image: {}'.format(frame_index, img_file_name));

        #######################################################
        ## Show Detections
        #######################################################

        label_detected = False;
        det_csv_path = '';

        image_current_det = copy.copy(image_current);

        det_object_list = [];
        if det_data_available:

           # CSV name management
            det_csv_name = img_file_name.split('.')[0] + '_det.csv';
            det_csv_path = os.path.join(args.det_dir_path, 'csv');
            det_csv_path = os.path.join(det_csv_path, det_csv_name);

            det_object_list = DetObject.from_csv(det_csv_path, expand_mask = True);

            for det_object in det_object_list:
                if det_object.label in args.label_list:
                    det_object.display_on_image(image_current_det);
                    label_detected = True;

        if label_detected or frame_index == 0 or frame_index == len(list_img_file) - 1 or step == 0:

            cv2.imshow('Detection', image_current_det)

            # Set callback to enable / disable detections
            cv2.setMouseCallback("Detection", click_detection, param=[det_object_list, det_csv_path, image_current, args.label_list, args.label_replace])


            wait_key = 0;

            key = cv2.waitKey(wait_key) & 0xFF

            if key == ord("q"):
                break;

            elif key == ord("n"):
                step = 1;

            elif key == ord("p"):
                step = -1;


        frame_index +=step;
        mathutil.clip(frame_index, 0, len(list_img_file));


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
