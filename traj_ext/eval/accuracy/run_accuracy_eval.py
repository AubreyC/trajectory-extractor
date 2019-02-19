
import numpy as np
import time
import cv2
import copy
from scipy.optimize import linear_sum_assignment
import sys
import argparse
import os
import configparser
from shutil import copyfile

import xml.etree.ElementTree as ET
import motmetrics

# Root directory of the project
FILE_PATH = os.path.abspath(os.path.dirname(__file__));
ROOT_DIR =  os.path.abspath(os.path.join(FILE_PATH,'../../'));
sys.path.append(ROOT_DIR);

import track_2D

from utils import cfgutil
from utils.mathutil import *

from object_det.mask_rcnn import detect_utils
from box3D_fitting import Box3D_utils
from camera_calib import calib_utils

from postprocess_track import trajutil
from postprocess_track import trajectory

from tracker import cameramodel as cm
from tracker.utils import tracker_utils

GROUND_TRUTH_PATH = 'data/auburn1_20171005_183440_accuracy/tracking_cvat/auburn1_20171005_183440_cvat.xml';
IMAGE_DATA_DIR = 'data/auburn1_20171005_183440_accuracy/img';
DET_DATA_DIR = 'data/auburn1_20171005_183440_accuracy/output/det/csv';
TRACK_DATA_DIR = 'data/auburn1_20171005_183440_accuracy/output/overlap_association/csv';
TRAJ_PATH = 'data/auburn1_20171005_183440_accuracy/output/smoothed/csv';
CAM_CFG_PATH = 'data/auburn1_20171005_183440_accuracy/auburn_camera_street_1_cfg.yml';
DET_ZONE_F_PATH = 'data/auburn1_20171005_183440_accuracy/auburn_camera_1_detection_zone.yml'

DELTA_MS = 67;
MAX_INDEX = 1000000;

LABEL_LIST = ['car', 'bus', 'truck'];

def get_tk_2D_in_list(tk_2D_list, id):

    for tk_2D in tk_2D_list:
        if id == tk_2D.get_id():
            return tk_2D;

    return None;

def plot_traj2D_on_images(track_2D_list, time_ms, img_street, cam_model_street, color = (0, 0, 255)):

    for track_2D in track_2D_list:

        # =========================================
        #               Plot filtering
        # ==========================================


        box2D = track_2D.get_box2D_at_timestamp(time_ms);
        if not (box2D is None):

            if not (img_street is None):

                pt_pix = (int(box2D.center[0]),int(box2D.center[1]));
                img_street = cv2.circle(img_street, pt_pix,3, (0, 0, 255), -1);

                # Add annotations to the track:
                text = "id: %i" % (track_2D.get_id());
                img_street = cv2.putText(img_street, text, pt_pix, cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 1)

    return img_street


def main():


    ##########################################################
    # Ground Truth trajectories
    ##########################################################
    track_2D_truth_list = [];
    tree = ET.parse(GROUND_TRUTH_PATH);
    root = tree.getroot();

    for data in root.iter('annotations'):

        for track in data.iter('track'):

            track_id = int(track.attrib['id']);
            label = track.attrib['label'];

            if label in LABEL_LIST:

                tk_init = True;

                for box in track.iter('box'):
                    fr_i = int(box.attrib['frame']);

                    occluded =  int(box.attrib['occluded']) == 1;

                    if not occluded:

                        if fr_i < MAX_INDEX:

                            if tk_init:
                                track_2D_tr = track_2D.Track_2D(track_id, label);
                                track_2D_truth_list.append(track_2D_tr);
                                print("Track id: {} label: {}".format(track.attrib['id'],track.attrib['label']))
                                tk_init = False;


                            time_ms = DELTA_MS*fr_i;
                            xtl = float(box.attrib['xtl']);
                            ytl = float(box.attrib['ytl']);
                            xbr = float(box.attrib['xbr']);
                            ybr = float(box.attrib['ybr']);

                            track_2D_tr.push_2Dbox_meas(time_ms, label, xtl, ytl, xbr, ybr);


    ##########################################################
    # Camera Parameters
    ##########################################################

    # Construct camera model
    cam_model_1 = calib_utils.read_camera_calibration(CAM_CFG_PATH);

    ##########################################################
    # Processed Trajetcories
    ##########################################################

    traj_list = trajutil.read_traj_list_from_csv(TRAJ_PATH);
    track_2D_list = [];
    for traj in traj_list:

        tk_2D_init = True;

        traj_point_list = traj.get_traj();
        for traj_point in traj_point_list:

            if traj_point.time_ms < MAX_INDEX*DELTA_MS:

                if tk_2D_init:
                    tk_2D = track_2D.Track_2D(traj.get_id(), None);
                    track_2D_list.append(tk_2D);
                    tk_2D_init = False;

                # Project 3Dbox corners on Image Plane
                pt = np.array([traj_point.x, traj_point.y, -1.0])
                pt.shape = (1,3);
                (pt_img, jacobian) = cv2.projectPoints(pt, cam_model_1.rot_CF_F, cam_model_1.trans_CF_F, cam_model_1.cam_matrix, cam_model_1.dist_coeffs)
                pt_img = (int(pt_img[0][0][0]), int(pt_img[0][0][1]));

                # pt_pix = cam_model_1.project_points(np.array([(tk.get_filt_pos(t_current)[0], tk.get_filt_pos(t_current)[1], 0.0)]));
                # pt_pix = (int(pt_pix[0]),int(pt_pix[1]));

                xtl = pt_img[0] - 5;
                ytl = pt_img[1] - 5;
                xbr = pt_img[0] + 5;
                ybr = pt_img[1] + 5;
                tk_2D.push_2Dbox_meas(traj_point.time_ms, None, xtl, ytl, xbr, ybr);


    fs_read = cv2.FileStorage(DET_ZONE_F_PATH, cv2.FILE_STORAGE_READ)
    pt_det_zone_FNED = fs_read.getNode('model_points_FNED').mat();
    pt_det_zone_pix = None;
    if not (pt_det_zone_FNED is None):
        pt_det_zone_pix = Box3D_utils.pt_det_zone_FNED_to_pix(pt_det_zone_FNED, cam_model_1)

    ##########################################################
    # Remove outside Detection ZONE
    ##########################################################
    for tk_2d in track_2D_list:
        for box2D in list(tk_2d._box2D_meas_list):

            if not (pt_det_zone_pix is None):

                in_zone_flag = Box3D_utils.in_detection_zone(box2D.roi, pt_det_zone_pix)
                if not in_zone_flag:
                    tk_2d._box2D_meas_list.remove(box2D);


    for tk_2d in track_2D_truth_list:
        for box2D in list(tk_2d._box2D_meas_list):

            if not (pt_det_zone_pix is None):

                in_zone_flag = Box3D_utils.in_detection_zone(box2D.roi, pt_det_zone_pix)
                if not in_zone_flag:
                    tk_2d._box2D_meas_list.remove(box2D);

    ##########################################################
    # Remove trajectories that are less than 1 seconds long
    ##########################################################
    for tr_2D in list(track_2D_list):
        l = tr_2D.get_length_ms();

        if l < 1000:
            track_2D_list.remove(tr_2D);
            print('Remove track_id: {} length:{} ms'.format(tr_2D.get_label(), l));


    for tr_2D in list(track_2D_truth_list):
        l = tr_2D.get_length_ms();

        if l < 1000:
            track_2D_truth_list.remove(tr_2D);
            print('Remove track_id: {} length:{} ms'.format(tr_2D.get_label(), l));

    ##########################################################
    # Compute metrics
    ##########################################################

    # Open img folder
    list_img_file = os.listdir(IMAGE_DATA_DIR);
    list_img_file.sort(key=lambda f: int(''.join(filter(str.isdigit, f))));

    time_ms_list = [];
    for current_index, current_img_file in enumerate(list_img_file):
        time_ms = current_index*DELTA_MS;
        time_ms_list.append(time_ms);


    acc = motmetrics.MOTAccumulator(auto_id=True)

    print('time_ms_list: {}'.format(len(time_ms_list)))
    for time_ms in time_ms_list:

        center_proc = [];
        id_proc = [];
        for track_2D_proc in track_2D_list:
            box2D = track_2D_proc.get_box2D_at_timestamp(time_ms);
            if not (box2D is None):
                center_proc.append(box2D.center);
                id_proc.append(track_2D_proc.get_id());


        center_truth = [];
        id_truth = [];
        for track_2D_truth in track_2D_truth_list:
            box2D = track_2D_truth.get_box2D_at_timestamp(time_ms);
            if not (box2D is None):
                center_truth.append(box2D.center);
                id_truth.append(track_2D_truth.get_id());

        dist_mat = np.zeros((len(id_truth), len(id_proc)));

        for index_proc in range(len(id_proc)):
            for index_truth in range(len(id_truth)):

                # Compute distance in pixels to match truth and hypothesis
                dist = np.linalg.norm(center_truth[index_truth]-center_proc[index_proc]);

                # Do not add distance if more than 50 pixels away
                if dist > 50:
                    dist = np.nan;
                dist_mat[index_truth, index_proc] = dist;

        acc.update(id_truth, id_proc, dist_mat);

    mh = motmetrics.metrics.create()

    summary = mh.compute( acc,\
                                metrics=motmetrics.metrics.motchallenge_metrics);

    strsummary = motmetrics.io.render_summary(\
        summary,\
        formatters=mh.formatters,\
        namemap=motmetrics.io.motchallenge_metric_names\
    )
    print(strsummary)

    # print(acc.events) # a pandas DataFrame containing all events
    summary = mh.compute(acc, metrics=['num_frames','mostly_tracked', 'mostly_lost', 'partially_tracked', 'num_switches','num_fragmentations'], name='acc')
    print(summary);

    print('Number truth: {}'.format(len(track_2D_truth_list)));
    print('Number detect: {}'.format(len(track_2D_list)));

    path_output_panda = 'MOT_panda_dataframe.csv';
    print('Saving dataframe output MOT metrics: {}'.format(path_output_panda))
    acc.events.to_csv(path_output_panda)

    print('\n\nShowing tracking on images:\n- Any key to move forward\n- q to exit\n\n')

    ##########################################################
    # Plot Results
    ##########################################################

    # print('list_img_file {}'.format(len(list_img_file)));
    for current_index, current_img_file in enumerate(list_img_file):

        time_ms = current_index*DELTA_MS;
        # time_ms = time_ms_list[current_index];

        if time_ms > MAX_INDEX*DELTA_MS:
            break;

        img_smooth = current_img_file;

        # Getting current street image
        img_street = cv2.imread(os.path.join(IMAGE_DATA_DIR, img_smooth));
        print('img_smooth: {}'.format(img_smooth))

        # Draw detection zone:
        if not (pt_det_zone_FNED is None):
            Box3D_utils.draw_det_zone(img_street, cam_model_1, pt_det_zone_FNED, color=(0,0,255), thickness=2);

        # Draw truth and detceted tracks:
        img_street = plot_traj2D_on_images(track_2D_list, time_ms, img_street, cam_model_1, color = (0,0,255)); # RED
        img_street = plot_traj2D_on_images(track_2D_truth_list, time_ms, img_street, cam_model_1, color = (255,0,0)); # BLUE

        cv2.imshow('img_street', img_street)

        # Normal Mode: Press q to exit
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break;

        print('[Info]: Plotting step: {}'.format(current_index))
        # time.sleep(0.1);


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')