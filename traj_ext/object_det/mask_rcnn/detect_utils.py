# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-04-05 09:50:35
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

import copy
import numpy as np
import csv

def encode_mask(mask, width, height):

    # Encode mask by just taking the contour of the masK
    # One dimension

    m = copy.copy(mask);

    # One dimension array
    m.shape = (1, m.size)

    # Padding: Add 0 to the left and right of the array (in order to be able to substarct shifted version)
    og_mask = np.append(np.zeros((1,1), int), m)
    og_mask = np.append(og_mask, np.zeros((1,1), int))

    # Starting
    og_mask_shifted_left = np.append(m, np.zeros((1,2), int))
    diff_start = np.subtract(og_mask, og_mask_shifted_left)
    # Shift in index due to padding cancel with the fact that the -1 appears at (index_start - 1)
    diff_index_start = np.where(diff_start == -1)[0];

    # Ending
    og_mask_shifted_right = np.append(np.zeros((1,2), int), m )
    diff_end = np.subtract(og_mask, og_mask_shifted_right)
    diff_index_end = np.where(diff_end == -1)[0];
    # Compensate for padding and the fact the -1 appears at (index_end + 1)
    diff_index_end = diff_index_end -2;

    diff_index = np.append(diff_index_start, diff_index_end);
    diff_index = np.sort(diff_index);
    return diff_index

def decode_mask_bool(diff_index, width, height):
    mask_new = np.zeros((1, width*height), np.bool);

    length = int(diff_index.size/2)
    for i in range(0, length):

        be = diff_index[2*i]
        end = diff_index[2*i+1]
        mask_new[0,be:end+1] = True;

    mask_new.shape = (height, width);
    return mask_new

def convert_str(diff_mask):
    b = ''.join(str(x) + ' ' for x in diff_mask)
    return  b.strip();

def back_str(b_str):
    a = b_str.split(' ')
    mask_bin = [int(x) for x in a];
    myarray = np.asarray(mask_bin, 'int')
    return myarray

def create_det_dict(r, class_names, det_ind):

    # Field management
    dict_det = {};
    dict_det['det_id'] = det_ind;
    dict_det['topleft_x'] = r['rois'][det_ind][1]
    dict_det['topleft_y'] = r['rois'][det_ind][0]
    dict_det['bottomright_x'] = r['rois'][det_ind][3]
    dict_det['bottomright_y'] = r['rois'][det_ind][2]
    dict_det['confidence'] = r['scores'][det_ind]

    #Label Name
    name_id = r['class_ids'][det_ind]
    name = class_names[name_id]
    dict_det['label'] = name

    # Mask: Convert in mask cont
    mask = r['masks'][:,:,det_ind]
    height = mask.shape[0]
    width = mask.shape[1]
    dict_det['height'] = height
    dict_det['width'] = width

    # Store mask
    mask.shape = (1,height*width)
    d = encode_mask(mask, width, height);
    mask_str = convert_str(d);
    dict_det['mask'] = mask_str;

    return dict_det;

def write_detection_csv(path_csv, r, class_names):

    csv_open = False;
    with open(path_csv, 'w') as csvfile:

        for index in range(0,len(r['rois'])):

            if not csv_open:
                fieldnames = [];

                # Add new keys
                fieldnames.append('det_id');
                fieldnames.append('topleft_x');
                fieldnames.append('topleft_y');
                fieldnames.append('bottomright_x');
                fieldnames.append('bottomright_y');
                fieldnames.append('confidence');
                fieldnames.append('label');
                fieldnames.append('height');
                fieldnames.append('width');
                fieldnames.append('mask');

                #Write field name
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
                writer.writeheader();

                # Flag to true
                csv_open = True;

            # Field management
            dict_det = create_det_dict(r, class_names, index);

            # Write detection in CSV
            if csv_open:
                writer.writerow(dict_det);


def read_dict_csv(csv_path):

    data_list = [];

    # Create dict
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:

            fields = row.keys();
            item = {};

            if 'det_id' in fields:
                item['det_id'] = int(row['det_id']);


            if 'width' in fields:
                item['width'] = int(row['width']);

            if 'width' in fields:
                item['width'] = int(row['width']);


            if 'height' in fields:
                item['height'] = int(row['height']);

            if 'confidence' in fields:
                item['confidence'] = np.float32(row['confidence']);

            if 'topleft_y' in fields:
                roi = np.array([row['topleft_y'], row['topleft_x'], row['bottomright_y'], row['bottomright_x']], dtype= np.int16);
                item['roi'] = roi;

            if 'label' in fields:
                item['label'] = row['label'];

            if 'mask' in fields:
                mask_str = row['mask'];

                width = int(row['width']);
                height = int(row['height']);

                mask_array = back_str(mask_str)
                mask = decode_mask_bool(mask_array, width, height);
                mask_array.shape = (int(mask_array.size/2), 2)
                mask.shape = (height, width)

                item['mask'] = mask;
                item['mask_cont'] = mask_array;

            if 'track_id' in fields:
                item['track_id'] = int(row['track_id']);

            if 'box_3D_phi' in fields:
                item['box_3D'] = [float(row['box_3D_phi']),\
                                  float(row['box_3D_x']),\
                                  float(row['box_3D_y']),\
                                  float(row['box_3D_z']),\
                                  float(row['box_3D_l']),\
                                  float(row['box_3D_w']),\
                                  float(row['box_3D_h'])];

            if 'percent_overlap' in fields:
                item['percent_overlap'] = float(row['percent_overlap']);

            data_list.append(item);

    return data_list

# Encode Bool mask into a str
def encode_mask_str(mask):

    mask_cop = copy.copy(mask);
    height = mask_cop.shape[0]
    width = mask_cop.shape[1]
    mask_cop.shape = (1,height*width)
    d = encode_mask(mask_cop, width, height);
    mask_str = convert_str(d);

    return mask_str;

def write_dict_csv(csv_path, data_list, class_names):

    # List of dict

    csv_open = False;
    with open(path_csv, 'w') as csvfile:

        for index in range(0,len(data_list)):

            if not csv_open:
                fieldnames = [];

                # Add new keys
                fieldnames.append('det_id');
                fieldnames.append('topleft_x');
                fieldnames.append('topleft_y');
                fieldnames.append('bottomright_x');
                fieldnames.append('bottomright_y');
                fieldnames.append('confidence');
                fieldnames.append('label');
                fieldnames.append('height');
                fieldnames.append('width');
                fieldnames.append('mask');
                fieldnames.append('track_id');
                fieldnames.append('box_3D_phi');
                fieldnames.append('box_3D_x');
                fieldnames.append('box_3D_y');
                fieldnames.append('box_3D_z');
                fieldnames.append('box_3D_l');
                fieldnames.append('box_3D_w');
                fieldnames.append('box_3D_h');

                #Write field name
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames);
                writer.writeheader();

                # Flag to true
                csv_open = True;

            # Field management
            dict_row = {};
            dict_row['det_id'] = data_list[index]['det_id'];

            # ROI:
            dict_row['topleft_x'] = data_list[index]['roi'][1];
            dict_row['topleft_y'] = data_list[index]['roi'][0];
            dict_row['bottomright_x'] = data_list[index]['roi'][3];
            dict_row['bottomright_y'] = data_list[index]['roi'][2];

            dict_row['confidence'] = data_list[index]['confidence'];
            dict_row['label'] = class_names[data_list[index]['class_id']];
            dict_row['height'] = data_list[index]['height'];
            dict_row['width'] = data_list[index]['width'];

            # Mask in str
            mask = data_list[index]['mask'];
            mask_str = encode_mask_str(mask);
            dict_row['mask'] = mask_str;

            #Track id:
            dict_row['track_id'] = data_list[index]['track_id'];

            # Raw box:
            dict_row['box_3D_phi'] = data_list[index]['box_3D'][0];
            dict_row['box_3D_x'] = data_list[index]['box_3D'][1];
            dict_row['box_3D_y'] = data_list[index]['box_3D'][2];
            dict_row['box_3D_z'] = data_list[index]['box_3D'][3];
            dict_row['box_3D_l'] = data_list[index]['box_3D'][4];
            dict_row['box_3D_w'] = data_list[index]['box_3D'][5];
            dict_row['box_3D_h'] = data_list[index]['box_3D'][6];

            # Write detection in CSV
            if csv_open:
                writer.writerow(dict_row);


def read_detection_csv(csv_path, class_names, only_mask_cont = False):
    # Create result dict
    r = dict.fromkeys(['scores', 'masks', 'rois', 'class_ids', 'mask_cont']);
    # r = dict.fromkeys(['scores', 'rois', 'class_ids', 'mask_cont']);

    init = False;

    # Create dict
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)

        for line in reader:

            width = int(line['width']);
            height = int(line['height']);

            score = np.float32(line['confidence']);
            roi = np.array([line['topleft_y'], line['topleft_x'], line['bottomright_y'], line['bottomright_x']], dtype= np.int16)
            roi.shape = (1,4)
            class_id = np.int16(class_names.index(line['label']));

            # Getting mask
            mask_str = line['mask'];
            mask_array = back_str(mask_str)

            # Do no compute mask if asked
            if not only_mask_cont:
                mask = decode_mask_bool(mask_array, width, height);
                mask.shape = ((height, width, 1))

            mask_array.shape = (int(mask_array.size/2), 2)

            if not init:
                scores = np.array(score, dtype= np.float32);
                scores.shape = (1,)
                rois = np.array(roi, dtype= np.int16);
                class_ids = np.array(class_id, dtype= np.int16);
                class_ids.shape = (1,)
                mask_cont = [mask_array];

                if not only_mask_cont:
                    masks = np.array(mask,  dtype= np.bool);

                init = True;

            else:
                scores = np.append(scores, np.float32(line['confidence']));
                rois = np.append(rois, roi, axis=0);
                class_ids = np.append(class_ids, class_id);
                mask_cont.append(mask_array);

                if not only_mask_cont:
                    masks = np.append(masks, mask, axis=2);


    if init:
        # Make the dict
        r['scores'] = scores;

        if not only_mask_cont:
            r['masks'] = masks;

        r['rois'] = rois;
        r['class_ids'] = class_ids;
        r['mask_cont'] = mask_cont

    else:
        r = None;

    return r


# if __name__ == '__main__':

#     # Just to test

#     # Create dummy mask:
#     width = 1280;
#     height = 720;
#     mask = np.zeros((height, width), np.bool);

#     mask[0,0] = True;
#     mask[16:85, 45:450] = True;

#     # Encode mask into a index of where it starts / ends and Convert index into a string
#     d = encode_mask(mask, width, height);
#     mask_str = convert_str(d);

#     # Decode
#     mask_array = back_str(mask_str)
#     m = decode_mask_bool(mask_array, width, height);

#     # Verification
#     print(np.array_equal(m, mask))

#     class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                    'bus', 'train', 'truck', 'boat', 'traffic light',
#                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                    'teddy bear', 'hair drier', 'toothbrush'];

    #r = read_detection_csv('auburn/det/auburn_1_test_11933.csv', class_names);
    #print(r)
