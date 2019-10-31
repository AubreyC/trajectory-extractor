# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-21 18:23:24
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-16 07:44:08


from traj_ext.postprocess_track import run_postprocess
from traj_ext.utils import cfgutil

def test_run_postprocess_tracker():

    # Read config file:
    path_config = 'traj_ext/postprocess_track/test/tracker_postprocess_cfg_test.json';

    # Run the det association:
    assert run_postprocess.main(['-config_json', path_config]);


if __name__ == '__main__':

    test_run_postprocess_tracker();