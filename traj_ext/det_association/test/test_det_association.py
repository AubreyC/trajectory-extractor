# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-21 18:23:24
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-15 22:38:46


from traj_ext.det_association import run_det_association
from traj_ext.utils import cfgutil

def test_run_det_association():

    # Read config file:
    path_config = 'traj_ext/det_association/test/det_association_cfg_test.json';

    # Run the det association:
    assert run_det_association.main(['-config_json', path_config]);

if __name__ == '__main__':

    test_run_det_association();