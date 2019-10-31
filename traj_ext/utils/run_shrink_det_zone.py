
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

from traj_ext.visualization import run_inspect_traj
from traj_ext.hd_map.HD_map import HDmap


from traj_ext.utils import det_zone


def main():

     # Arg parser:
    parser = argparse.ArgumentParser(description='Visualize the trajectories');
    parser.add_argument('-det_zone', dest="det_zone_path", type=str, help='Path to the detection zone yml', default = 'traj_ext/camera_calib/calib_file/brest/brest_area1_detection_zone.yml');
    parser.add_argument('-shrink_zone', dest="shrink_zone", type=float, help='Detection zone shrink coefficient for complete trajectories', default =1);
    parser.add_argument('-camera', dest="cam_model_path", type=str, help='Path of the camera model yml', default='traj_ext/camera_calib/calib_file/brest/brest_area1_street_cfg.yml');
    parser.add_argument('-image', dest="image_path", type=str, help='Path of the image', default = 'traj_ext/camera_calib/calib_file/brest/brest_area1_street.jpg');

    args = parser.parse_args()

    # Read from yml
    det_zone_FNED = det_zone.DetZoneFNED.read_from_yml(args.det_zone_path);

    # Srhrink
    det_zone_FNED_shrinked = det_zone_FNED.shrink_zone(args.shrink_zone);

    # Get image
    image_current = cv2.imread(args.image_path);

    # Read cam model
    cam_model = CameraModel.read_from_yml(args.cam_model_path);

    save_flag = False;
    while True:


        image_current = det_zone_FNED.display_on_image(image_current, cam_model, thickness = 1);

        image_current = det_zone_FNED_shrinked.display_on_image(image_current, cam_model, thickness = 1);

        cv2.imshow('Image', image_current);


        key = cv2.waitKey(0) & 0xFF

        if key == 13:
            save_flag = True;
            break;

        elif key == ord("q"):
            break;


    if save_flag:
        # Get path:
        det_zone_FNED_shrinked_path = args.det_zone_path.replace('.yml', '_shrinked_' + ('%i' % (args.shrink_zone*100)) + '.yml');

        # Save to yml
        det_zone_FNED_shrinked.save_to_yml(det_zone_FNED_shrinked_path)

        # Create det_zone_image
        det_zone_im_shrinked = det_zone_FNED_shrinked.create_det_zone_image(cam_model);

        det_zone_im_shrinked_path = args.det_zone_path.replace('.yml', '_shrinked_' + ('%i' % (args.shrink_zone*100)) + '_im.yml');

        det_zone_im_shrinked.save_to_yml(det_zone_im_shrinked_path)

if __name__ == '__main__':

    main()