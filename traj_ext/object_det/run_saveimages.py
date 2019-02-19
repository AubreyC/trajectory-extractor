# -*- coding: utf-8 -*-

###################################################################################
#
# Script that opens a video and save the frames individually with their timestamps
#
###################################################################################

import cv2
import numpy as np
import time
import copy
import os
import csv
import argparse

if __name__ == '__main__':

    ##########################################################
    # Cmd Line program
    ##########################################################

    parser = argparse.ArgumentParser(description='Split video into timestamped images. Name are "videoname_XXXXXX.jpg" with XXXXX timestamp in ms.');
    parser.add_argument(dest="video_path", type=str, help='path of the video_file' );
    parser.add_argument('-o', dest="output_folder", type=str, help='path of the output folder' );
    args = parser.parse_args()

    ##########################################################
    # Open Video
    ##########################################################

    # Init the video reader
    capture = cv2.VideoCapture(args.video_path)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ##########################################################
    # Open Output files
    ##########################################################

    # Get name of the video
    video_name = args.video_path.split('/')[-1];
    video_name = video_name.split('.')[0];

    # If outpath is not specified, svae images in the output folder
    if args.output_folder is None:

        # Create output folder if necessary
        output_path = 'output/' + video_name;

    else:
        output_path = args.output_folder;

    output_path_img = output_path + '/img'
    os.makedirs(output_path_img, exist_ok=True)

    print('Saving frames:{} '.format(output_path_img))

    ##########################################################
    # Run through the video
    ##########################################################

    # Record frame number
    frame_number = 0;

    save = True;
    ret = True;
    while ret:

        #Get time
        stime = time.time()
        ret, frame = capture.read()
        time_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC));
        time_ms_str = str(time_ms).zfill(10);
        # Save video
        current_name = video_name + '_' + time_ms_str;
        name_image = current_name + '.jpg';

        if save:

            cv2.imwrite( output_path_img + '/' + name_image, frame );
            print('Frame Number: {} {}'.format(frame_number, name_image))
            save = False;
        else:
            save = True;

        frame_number = frame_number + 1;

    print('\n**** Job is done ****\n{} frames saved at {}\n'.format(frame_number, output_path_img))

    capture.release()
    cv2.destroyAllWindows()