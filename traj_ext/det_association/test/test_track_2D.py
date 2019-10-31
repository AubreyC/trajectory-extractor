# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-21 18:23:24
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-27 22:35:51

import os
import cv2

from traj_ext.det_association import track_2D
from traj_ext.utils import cfgutil

# Define output dir for the test
DIR_PATH = os.path.dirname(__file__);
OUTPUT_DIR_PATH = os.path.join(DIR_PATH,'test_output');


def test_read_track_2D(show_images =False):

    # Read config file:
    img_folder_path = 'test_dataset/brest_20190609_130424_327_334/img';
    det_folder_path = 'test_dataset/brest_20190609_130424_327_334/output/det/csv';
    det_asso_folder_path = 'test_dataset/brest_20190609_130424_327_334/output/vehicles/det_association/csv';


    list_img_file = os.listdir(img_folder_path);
    list_img_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    list_img_file = list_img_file[:200];

    # Run the det association:
    track_2D_list, frame_index_list = track_2D.Track2D.from_csv(list_img_file, det_folder_path, det_asso_folder_path);

    # Create outut dir if not created
    det_asso_folder_path_test = os.path.join(OUTPUT_DIR_PATH, 'det_association/test_track_2D')
    os.makedirs(det_asso_folder_path_test, exist_ok=True)
    print('det_asso_folder_path_test:{}'.format(det_asso_folder_path_test))

    track_2D.Track2D.export_det_asso_csv(list_img_file, track_2D_list, det_asso_folder_path_test);

    # Show image
    if show_images:

        for frame_index, image_name in enumerate(list_img_file):
            cv_image = cv2.imread(os.path.join(img_folder_path, image_name));

            if not (cv_image is None):

                for tk_2D in track_2D_list:

                    det_object = tk_2D.get_det_frame_index(frame_index);
                    if not (det_object is None):
                        det_object.display_on_image(cv_image, track_id_text = True, color = tk_2D.color);


                cv2.imshow('frame', cv_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    return True;

if __name__ == '__main__':

    test_read_track_2D(show_images = True);