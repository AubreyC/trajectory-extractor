# import the necessary packages
import argparse
import cv2
import sys
import os
import numpy as np
import time
import copy
from scipy.optimize import linear_sum_assignment
import configparser

from traj_ext.tracker import cameramodel
from traj_ext.utils import det_zone

from traj_ext.hd_map.HD_map import HDmap

def click(event, x, y, flags, param ):

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed

    pt_image_list = param[0]
    cam_model = param[1];
    image = param[2];
    det_zone_FNED = param[3];
    hd_map = param[4];
    road_mark_id = param[5][0];
    road_mark_type = param[6][0];
    cam_model_sat = param[7];
    image_sat = param[8];

    if event == cv2.EVENT_LBUTTONDOWN:
        pt_image = (x, y)

        add_point_HD_map(cam_model, hd_map, road_mark_id, road_mark_type, pt_image);

        draw_image(image, "image_camera", hd_map, cam_model);
        draw_image(image_sat, "image_sat", hd_map, cam_model_sat);

    return;


def add_point_HD_map(cam_model, hd_map, road_mark_id, road_mark_type, pt_image):

    if not (cam_model is None):

            pos_FNED = cam_model.projection_ground(0, pt_image);
            pos_FNED.shape = (3,1);
            pos_FNED = pos_FNED[0:2,:];

            hd_map.add_point(road_mark_id, road_mark_type, pos_FNED);



def draw_image(image, image_name, hd_map, cam_model):

        image_temp = copy.copy(image);

        image_temp = hd_map.display_on_image(image_temp, cam_model, show_number = True);
        cv2.imshow(image_name, image_temp);

def main():

    # ##########################################################
    # # Parse Arguments
    # ##########################################################

    parser = argparse.ArgumentParser(description='Draw HD maps on image');
    parser.add_argument('-image', dest="image_path", type=str, help='Path of the image', default='traj_ext/camera_calib/calib_file/biloxi/biloxi_cam.jpg');
    parser.add_argument('-camera', dest="cam_model_path", type=str, help='Path of the satellite camera model yml', default='traj_ext/camera_calib/calib_file/biloxi/biloxi_cam_cfg.yml');
    parser.add_argument('-detection_zone', dest="detection_zone_path", type=str, help='Path of the detection zone yml', default='');
    parser.add_argument('-origin_lat', dest="origin_lat", type=str, help='Latitude of the origin', default='0.0');
    parser.add_argument('-origin_lon', dest="origin_lon", type=str, help='Longitude of the origin', default='0.0');
    parser.add_argument('-hd_map', dest="hd_map_path", type=str, help='Path to the HD map', default='');
    parser.add_argument('-output_folder', dest="output_folder_path", type=str, help='Path of the folder to save HD map', default='');

    args = parser.parse_args();

    # Print instructions
    print("############################################################")
    print("Generate HD map")
    print("############################################################\n")

    # Construct camera model
    cam_model = None;

    if args.cam_model_path != '':
        cam_model = cameramodel.CameraModel.read_from_yml(args.cam_model_path);

    # load the image, clone it, and setup the mouse callback function
    image_1 = cv2.imread(args.image_path)
    if image_1 is None:
        print('\n[Error]: image_path is not valid: {}'.format(args.image_path));
        return;

    # Get origin lat/lon
    origin_latlon = np.array([float(args.origin_lat), float(args.origin_lon)]);

    # Det zone:
    det_zone_FNED = None;
    if args.detection_zone_path:
        det_zone_FNED = det_zone.DetZoneFNED.read_from_yml(args.detection_zone_path);

    # Create windows
    cv2.namedWindow("image_camera")
    cv2.namedWindow("image_sat")

    name = args.image_path.split('/')[-1].split('.')[0];

    if os.path.isfile(args.hd_map_path):
        hd_map = HDmap.from_csv(args.hd_map_path);
    else:
        hd_map = HDmap(name, origin_latlon);


    # create image sat
    cam_model_sat, image_sat = hd_map.create_view( dist_max = 80.0);

    road_mark_id_list = [len(hd_map.road_marks)];
    road_mark_type_list = [hd_map.TYPE_CURB];
    pt_image_list = [];
    cv2.setMouseCallback("image_camera", click, param=(pt_image_list, cam_model, image_1, det_zone_FNED, hd_map, road_mark_id_list, road_mark_type_list, cam_model_sat, image_sat))

    draw_image(image_1, "image_camera", hd_map, cam_model);
    draw_image(image_sat, "image_sat", hd_map, cam_model_sat);


    # keep looping until the 'q' key is pressed
    save = False;
    while True:

        # display the image and wait for a keypress
        key = cv2.waitKey(0) & 0xFF
        # if key == ord("c"):
        # if the 'Enter' key is pressed, end of the program
        if key == 13:
            save = True;
            break

        # Esc exit
        elif  key == 27:
            break;

        elif key == ord("n"):
            road_mark_id_list[0] = len(hd_map.road_marks);
            print('New Road Mark: {}'.format(road_mark_id_list[0]));

            draw_image(image_1, "image_camera", hd_map, cam_model);
            draw_image(image_sat, "image_sat", hd_map, cam_model_sat);


        elif key == ord("d"):
            hd_map.delete_last_road_mark(road_mark_id_list[0]);

            # road_mark_id_list[0] = max(0, (len(hd_map.road_marks) - 1));

            draw_image(image_1, "image_camera", hd_map, cam_model);
            draw_image(image_sat, "image_sat", hd_map, cam_model_sat);


        elif key == ord("z"):

            hd_map.add_xy_offset(-0.1, 0.0);
            draw_image(image_1, "image_camera", hd_map, cam_model);
            draw_image(image_sat, "image_sat", hd_map, cam_model_sat);

        elif key == ord("c"):

            hd_map.add_xy_offset(+0.1, 0.0);
            draw_image(image_1, "image_camera", hd_map, cam_model);
            draw_image(image_sat, "image_sat", hd_map, cam_model_sat);

        elif key == ord("s"):

            hd_map.add_xy_offset(0.0, +0.1);
            draw_image(image_1, "image_camera", hd_map, cam_model);
            draw_image(image_sat, "image_sat", hd_map, cam_model_sat);

        elif key == ord("x"):

            hd_map.add_xy_offset(0.0, -0.1);
            draw_image(image_1, "image_camera", hd_map, cam_model);
            draw_image(image_sat, "image_sat", hd_map, cam_model_sat);

        elif key == ord("+"):

            road_mark_id_list[0] = min((len(hd_map.road_marks) - 1), (road_mark_id_list[0] + 1));
            print('Selecting Road Mark: {}'.format(road_mark_id_list[0]));

        elif key == ord("-"):

            road_mark_id_list[0] = max(0, (road_mark_id_list[0] - 1));
            print('Selecting Road Mark: {}'.format(road_mark_id_list[0]));

        elif key == ord(str(1)):
            road_mark_type_list[0] = HDmap.TYPE_CURB;

            hd_map.set_road_mark_type(road_mark_id_list[0], road_mark_type_list[0]);

            draw_image(image_1, "image_camera", hd_map, cam_model);
            draw_image(image_sat, "image_sat", hd_map, cam_model_sat);

        elif key == ord(str(2)):
            road_mark_type_list[0] = HDmap.TYPE_LANE_PLAIN;

            hd_map.set_road_mark_type(road_mark_id_list[0], road_mark_type_list[0]);

            draw_image(image_1, "image_camera", hd_map, cam_model);
            draw_image(image_sat, "image_sat", hd_map, cam_model_sat);

        elif key == ord(str(3)):
            road_mark_type_list[0] = HDmap.TYPE_LANE_DOTTED;

            hd_map.set_road_mark_type(road_mark_id_list[0], road_mark_type_list[0]);

            draw_image(image_1, "image_camera", hd_map, cam_model);
            draw_image(image_sat, "image_sat", hd_map, cam_model_sat);

        elif key == ord(str(4)):
            road_mark_type_list[0] = HDmap.TYPE_STOP_PLAIN;

            hd_map.set_road_mark_type(road_mark_id_list[0], road_mark_type_list[0]);

            draw_image(image_1, "image_camera", hd_map, cam_model);
            draw_image(image_sat, "image_sat", hd_map, cam_model_sat);


        elif key == ord(str(5)):
            road_mark_type_list[0] = HDmap.TYPE_STOP_DOTTED;

            hd_map.set_road_mark_type(road_mark_id_list[0], road_mark_type_list[0]);

            draw_image(image_1, "image_camera", hd_map, cam_model);
            draw_image(image_sat, "image_sat", hd_map, cam_model_sat);

        else:

            print('\nInstruction:\n- n: New Road Mark\
                                 \n- d: Delete last point\
                                 \n- 1: Change type: TYPE_CURB\
                                 \n- 2: Change type: TYPE_LANE_PLAIN\
                                 \n- 3: Change type: TYPE_LANE_DOTTED\
                                 \n- 4: Change type: TYPE_STOP_PLAIN\
                                 \n- 5: Change type: TYPE_STOP_DOTTED\
                                 \n- +: Change selected road mark (next)\
                                 \n- -: Change selected road mark (previous)\
                                 \n- z: Shift all map -0.1 North\
                                 \n- c: Shift all map +0.1 North\
                                 \n- x: Shift all map -0.1 East\
                                 \n- s: Shift all map +0.1 East\
                                 \n- enter: Save HD map\
                                 \n- esc: Quit\
                                 \n')

    if save:

        # Save next to image sat or to the folder output
        csv_path = os.path.join(os.path.dirname(args.image_path), name + '_hd_map.csv');

        if os.path.isdir(args.output_folder_path):
            csv_path = os.path.join(args.output_folder_path, name +'_hd_map.csv');

        hd_map.to_csv(csv_path);


    print("Program Exit\n")
    print("############################################################\n")

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
