
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

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../../'));
sys.path.append(ROOT_DIR);

from object_det.mask_rcnn import detect_utils
from utils import cfgutil

# Import Mask RCNN
ROOT_DIR_MASKRCNN = os.path.join(FILE_PATH, "Mask_RCNN")
sys.path.append(ROOT_DIR_MASKRCNN)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
from samples.coco import coco

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

# Create a default cfg file which holds default values for the path
def create_default_cfg():

    config = configparser.ConfigParser();

    config['INPUT_PATH'] = \
                        {'IMAGE_DATA_DIR': ''}

    config['OUTPUT_PATH'] = \
                        {'OUTPUT_DIR': ''}

    config['OPTIONS'] = \
                        {'SAVE_CSV' : 1, \
                         'SHOW_IMAGES' : 0 }


    # Header of the cfg file
    text = '\
##########################################################################################\n\
#\n\
# OBJECT DETECTOR WITH MASK-RCNN\n\
#\n\
# Detect object in successive frames using Mask-RCNN\n\
#\n\
# Please modify this config file according to your configuration.\n\
# Path must be ABSOLUTE PATH\n\
##########################################################################################\n\n'

    # Write the cfg file
    with open('DETECTOR_MRCNN_CFG.ini', 'w') as configfile:
        configfile.write(text);
        config.write(configfile)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def main():

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
        '-i', '--init',
        action='store_true',
        help='Generate config files to fill in order to run the object detector: DETECTOR_MRCNN_CFG.ini')
    argparser.add_argument(
        '-cfg', '--config',
        default='DETECTOR_MRCNN_CFG.ini',
        help='Path to the config file')
    args = argparser.parse_args();


    # In init mode
    if args.init:

        # Create default config file
        create_default_cfg();

        print('Please fill the config files and restart the program:\n-DETECTOR_MRCNN_CFG.ini')

        return;

    # ##########################################################
    # # Read config file:
    # ##########################################################
    config = cfgutil.read_cfg(args.config);

    # Option for output
    save_csv = bool(float(config['OPTIONS']['SAVE_CSV']) != '0')
    show_images = bool(float(config['OPTIONS']['SHOW_IMAGES'] != '0'))

    # Directory of images to run detection on
    IMAGE_DIR = config['INPUT_PATH']['IMAGE_DATA_DIR'];
    if not os.path.exists(IMAGE_DIR):
      print('[Error]: IMAGE_DATA_DIR does not exist: {}'.format(config['INPUT_PATH']['IMAGE_DATA_DIR']));
      return;

    OUTPUT_DIR = config['OUTPUT_PATH']['OUTPUT_DIR']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # if SAVE_IMAGES:
    #     OUTPUT_IMG = os.path.join(OUTPUT_DIR, "img");
    #     os.makedirs(OUTPUT_IMG, exist_ok=True)

    OUTPUT_DET = os.path.join(OUTPUT_DIR, "csv");
    os.makedirs(OUTPUT_DET, exist_ok=True)

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
    list_file = os.listdir(IMAGE_DIR);

    try:
        list_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))));
    except Exception as e:
        print(e);

    for file_name in list_file:

        print('Detection for image {}'.format(file_name));

        # Open Image
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

        # Reshape image to have to be dim = 3  - Sometimes images have alpha as 4th dim
        if (image.shape[2] == 4):
            image = image[:,:,:-1]

        #Run detection
        results = model.detect([image], verbose=1)
        r = results[0]

        # Save Detections in csv
        if save_csv:
            name_csv = file_name.split('.')[0] + '_det.csv';
            detect_utils.write_detection_csv(os.path.join(OUTPUT_DET, name_csv), r, class_names);

        # Show images
        if show_images:
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
