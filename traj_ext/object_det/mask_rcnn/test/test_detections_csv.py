# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-21 18:23:24
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-08-19 16:02:45


from traj_ext.object_det.mask_rcnn import run_detections_csv
from traj_ext.utils import cfgutil
import json

def test_run_detections_csv():

    # Read config file:
    path_config = 'traj_ext/object_det/mask_rcnn/test/detector_maskrcnn_cfg_test.json';

    # Run the det association:
    run_detections_csv.main(['-config_json', path_config, '-frame_limit', '5']);

if __name__ == '__main__':

    test_run_detections_csv();