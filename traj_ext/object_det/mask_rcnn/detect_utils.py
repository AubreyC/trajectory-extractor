# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-04-05 09:50:35
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

import copy
import numpy as np
import csv


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