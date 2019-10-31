# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-02-08 16:00:02
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:


import numpy as np
import pytest

from traj_ext.utils import mathutil


def test_latlon_to_NED():

    latlon_1_21 = np.array([32.606561,-85.482366]);
    latlon_2_21 = np.array([32.606395, -85.481866 ]);
    latlon_3_21 = np.array([32.606558, -85.481720 ]);
    latlon_4_21 = np.array([32.606403, -85.481687 ]);


    # Test latlon -> NED -> latlon
    ned = mathutil.latlon_to_NED(latlon_1_21, latlon_1_21);
    latlon_2_21_test = mathutil.NED_to_latlon(latlon_1_21, ned);

    assert (np.linalg.norm((latlon_2_21 - latlon_2_21_test)) < 0.001);

    print ('NED: 1')
    print (ned);

    ned = mathutil.latlon_to_NED(latlon_1_21, latlon_2_21);
    print ('NED: 2')
    print (ned);

    ned = mathutil.latlon_to_NED(latlon_1_21, latlon_3_21);
    print ('NED: 3')
    print (ned);

    ned = mathutil.latlon_to_NED(latlon_1_21, latlon_4_21);
    print ('NED: 4')
    print (ned);



def test_euler_rotation_matrix():

    eulers_init = np.array([np.deg2rad(60), np.deg2rad(54) , np.deg2rad(-20)]);
    rot_CF_F = mathutil.eulerAnglesToRotationMatrix(eulers_init);
    eulers_dest = mathutil.rotationMatrixToEulerAngles(rot_CF_F);

    # print('eulers_init:');
    # print(eulers_init);
    # print('eulers_dest:');
    # print(eulers_dest);

    assert (np.allclose(eulers_init, eulers_dest))

def test_quaternion_rotation_matrix():

    # Test Quaternion to Rotation
    eulers_init = [np.deg2rad(60), np.deg2rad(54) , np.deg2rad(-20)];
    rot_CF_F = mathutil.eulerAnglesToRotationMatrix(eulers_init);
    q_CF_F = mathutil.rotationMatrixToQuaternion(rot_CF_F);
    rot_CF_F_compu = mathutil.quaternionsToRotationMatrix(q_CF_F);


    # print('q_CF_F:')
    # print(q_CF_F)
    # print('rot_CF_F')
    # print(rot_CF_F)
    # print('rot_CF_F_compu')
    # print(rot_CF_F - rot_CF_F_compu)

    assert (np.allclose(rot_CF_F, rot_CF_F_compu))

def test_compute_angle_diff(verbose = False):

    phi_i = +np.pi - 0.1;
    phi_f = -np.pi + 0.2;

    phi_diff = mathutil.compute_angle_diff(phi_f, phi_i);

    if verbose:
        print('compute_angle_diff: phi_f:{} phi_i:{} phi_diff:{}'.format(phi_f, phi_i, phi_diff));

    assert (phi_diff == pytest.approx(0.3))

if __name__ == '__main__':

    try:
        test_latlon_to_NED();
        test_euler_rotation_matrix();
        test_quaternion_rotation_matrix();
        test_compute_angle_diff(verbose = True);

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

