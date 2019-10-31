# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2019-05-21 18:23:24
# @Last Modified by:   Aubrey
# @Last Modified time: 2019-10-16 07:48:22


from traj_ext.box3D_fitting import run_optim_3Dbox_mono
from traj_ext.utils import cfgutil

def test_optim_3Dbox_mono():

    # Read config file:
    path_config = 'traj_ext/box3D_fitting/test/optim_box3D_mono_test.json';

    # Run the det association:
    assert run_optim_3Dbox_mono.main(['-config_json', path_config]);

if __name__ == '__main__':

    test_optim_3Dbox_mono();