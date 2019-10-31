
# -*- coding: utf-8 -*-

##########################################################################################
#
# OBJECT DETECTOR WITH MASK-RCNN
#
# Run Mask-RCNN on images from an image folder and save the results in csv files.
# It uses the pre-trained Mask-RCNN model on COCO to detect and segment objects.
##########################################################################################

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import configparser
import argparse
from shutil import copyfile
import json

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
# ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../../'));
# sys.path.append(ROOT_DIR);

from traj_ext.object_det.mask_rcnn import detect_utils
from traj_ext.utils import cfgutil

# Import Mask RCNN
ROOT_DIR_MASKRCNN = os.path.join(FILE_PATH, "Mask_RCNN")
sys.path.append(ROOT_DIR_MASKRCNN)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
from samples.coco import coco

from traj_ext.object_det.det_object import DetObject
import cv2

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def main(args_input):

    # Print instructions
    print("############################################################")
    print("Object Detector with Mask-RCNN")
    print("############################################################\n")

    # ##########################################################
    # # Parse Arguments
    # ##########################################################
    argparser = argparse.ArgumentParser(
        description='Object Detector with Mask-RCNN')
    argparser.add_argument(
        '-image_dir',
        default='',
        help='Path of the image folder')
    argparser.add_argument(
        '-output_dir',
        default='',
        help='Path of the output');
    argparser.add_argument(
        '-no_save_csv',
        action ='store_true',
        help='Do not save output as csv');
    argparser.add_argument(
        '-no_save_images',
        action ='store_true',
        help='Do not save output images');
    argparser.add_argument(
        '-show_images',
        action ='store_true',
        help='Show detections on images');
    argparser.add_argument(
        '-crop_x1y1x2y2',
        nargs = '*',
        type=int,
        default =[],
        help='Crop region: X1 Y1 X2 Y2');
    argparser.add_argument(
        '-config_json',
        default='',
        help='Path to json config')
    argparser.add_argument(
        '-frame_limit',
        type=int,
        default=0,
        help='Frame limit: 0 = no limit')

    args = argparser.parse_args(args_input);

    if os.path.isfile(args.config_json):
        with open(args.config_json, 'r') as f:
            data_json = json.load(f)
            vars(args).update(data_json)

    vars(args).pop('config_json', None);

    run_detections_csv(args);

def run_detections_csv(config):

    # Create output folder
    output_dir = config.output_dir;
    output_dir = os.path.join(output_dir, 'det');
    os.makedirs(output_dir, exist_ok=True)

    # Save the cfg file with the output:
    try:
        cfg_save_path = os.path.join(output_dir, 'detector_maskrcnn_cfg.json');
        with open(cfg_save_path, 'w') as json_file:
            config_dict = vars(config);
            json.dump(config_dict, json_file, indent=4)
    except Exception as e:
        print('[ERROR]: Error saving config file in output folder:\n')
        print('{}'.format(e))
        return;

    # Option for output
    save_csv = not config.no_save_csv;
    save_images = not config.no_save_images;
    show_images = config.show_images
    frame_limit = config.frame_limit;

    crop_x1y1x2y2 = config.crop_x1y1x2y2;
    crop_mode = False;
    if sum(crop_x1y1x2y2) > 0 and len(crop_x1y1x2y2) == 4:
        crop_mode = True;

    # Directory of images to run detection on
    image_dir = config.image_dir;
    if not os.path.exists(image_dir):
      print('[Error]: image_dir does not exist: {}'.format(image_dir));
      return;

    if save_images:
        output_img = os.path.join(output_dir, "img");
        os.makedirs(output_img, exist_ok=True)

    if save_csv:
        output_det = os.path.join(output_dir, "csv");
        os.makedirs(output_det, exist_ok=True)

    # ##########################################################
    # #  R-CNN Network configuration
    # ##########################################################

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR_MASKRCNN, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR_MASKRCNN, "mask_rcnn_coco.h5")

    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
    config_inf = InferenceConfig()
    config_inf.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_inf)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Load images from the images folder sorted by name
    list_file = os.listdir(image_dir);

    try:
        list_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))));
    except Exception as e:
        print(e);

    for frame_index, file_name in enumerate(list_file):

        #Check for frame limit:
        if frame_limit > 0:
            if frame_index > frame_limit:
                print("Reached frame limit: {}\n".format(frame_limit));
                break;

        print('Detection for image {}'.format(file_name));

        # Open Image
        image = skimage.io.imread(os.path.join(image_dir, file_name))

        # Reshape image to have to be dim = 3  - Sometimes images have alpha as 4th dim
        if (image.shape[2] == 4):
            image = image[:,:,:-1]

        # Crop image
        if crop_mode:
            img_height_og = image.shape[0];
            img_width_og = image.shape[1];

            image = image[crop_x1y1x2y2[1]:crop_x1y1x2y2[3], crop_x1y1x2y2[0]:crop_x1y1x2y2[2], :];

        #Run detection
        results = model.detect([image], verbose=1)
        r = results[0]

        # List of result
        det_object_list = [];
        for det_id in range(0,len(r['rois'])):

            # Extract info
            det_2Dbox = np.array([r['rois'][det_id][0], r['rois'][det_id][1], r['rois'][det_id][2], r['rois'][det_id][3]], dtype= np.int16);
            det_2Dbox.shape = (4,1);

            confidence = r['scores'][det_id]

            label_id = r['class_ids'][det_id]
            label = class_names[label_id]

            det_mask = r['masks'][:,:,det_id]
            height = det_mask.shape[0]
            width = det_mask.shape[1]

            # Create det object
            det_object = DetObject(det_id, label, det_2Dbox, confidence, image_width = width, image_height = height, det_mask = det_mask, frame_name = file_name, frame_id = frame_index);

            if crop_mode:
                det_object = det_object.from_cropped_image(crop_x1y1x2y2[0], crop_x1y1x2y2[1], crop_x1y1x2y2[2], crop_x1y1x2y2[3], img_width_og, img_height_og)

            det_object_list.append(det_object);

        # Save Detections in csv
        if save_csv:

            # Create name
            name_csv = file_name.split('.')[0] + '_det.csv';
            csv_path = os.path.join(output_det, name_csv);

            # Save detections in csv:
            DetObject.to_csv(csv_path, det_object_list);

        # Show images
        if show_images or save_images:

            image = cv2.imread(os.path.join(image_dir, file_name));

            # Crop image
            if crop_mode:
                tl = (crop_x1y1x2y2[0], crop_x1y1x2y2[1])
                br = (crop_x1y1x2y2[2], crop_x1y1x2y2[3])

                # Display 2D bounding box
                image = cv2.rectangle(image, tl, br, (0,0,255), 1);

            for det_object in det_object_list:
                det_object.display_on_image(image);

            if save_images:
                name_png = file_name.split('.')[0] + '_det.png';
                cv2.imwrite( os.path.join(output_img, name_png), image );

            if show_images:
                cv2.imshow('Test Object Det', image)
                key = cv2.waitKey(0) & 0xFF;

                if key == ord('q'):
                    break;

if __name__ == '__main__':

    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
