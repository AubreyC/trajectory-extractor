# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-02-08 16:00:02
# @Last Modified by:   Aubrey
# @Email: clausse.aubrey@gmail.com
# @Github:

###############################################
#
# Math Util functions
#
###############################################

import math;
import numpy as np;

# Define constant:
LOCATION_SCALING_FACTOR = 111318.84502145034;
LOCATION_SCALING_FACTOR_INV = 0.000008983204953368922;


# Wrap angle into -pi, pi
def wraptopi(x):
    pi = np.pi
    x = x - np.floor(x/(2*pi)) *2 *pi
    if x >= pi:
        x = x- 2*pi
    return x

def compute_angle_diff(phi_f, phi_i):
    # Compute phi_f - phi_i:

    phi_f = wraptopi(phi_f);
    phi_i = wraptopi(phi_i);

    phi_diff = phi_f - phi_i;
    phi_diff = wraptopi(phi_diff);

    # Test
    phi_f_result = phi_i + phi_diff;
    phi_f_result = wraptopi(phi_f_result);
    phi_f = wraptopi(phi_f);

    if(abs(phi_f - phi_f_result) > 0.0001):
        print('compute_angle_diff: [Error] phi_f:{} phi_i:{} phi_diff:{}'.format(phi_f, phi_i, phi_diff));
        print('phi_f_result: {} phi_i:{}'.format(phi_f_result, phi_f))
        return None;

    return phi_diff;


# Conversion lat/lon to NED:
def latlon_to_NED(latlon_orig_21, latlon_dest_21):
    # From:
    # https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Math/location.h
    # https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Math/location.cpp
    north = float((latlon_dest_21[0] - latlon_orig_21[0])*LOCATION_SCALING_FACTOR);
    east = float((latlon_dest_21[1] - latlon_orig_21[1])*LOCATION_SCALING_FACTOR*longitude_scale(latlon_orig_21[0]));
    return np.array([north,east]);

# Conversion NED to lat/lon:
def NED_to_latlon(latlon_orig_21, pos_ned_21):
    # From:
    # https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Math/location.h
    # https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Math/location.cpp
    lat = latlon_orig_21[0] + float(pos_ned_21[0]*LOCATION_SCALING_FACTOR_INV);
    lon = latlon_orig_21[1] + float(pos_ned_21[1]*LOCATION_SCALING_FACTOR_INV)/longitude_scale(latlon_orig_21[0]);
    return np.array([lat, lon]);

def longitude_scale(lat):
    # From:
    # https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Math/location.h
    # https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Math/location.cpp
    scale = float(math.cos(math.radians(lat)))
    return scale;

## From: https://www.learnopencv.com/rotation-matrix-to-euler-angles/

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    # Because the follwing formular is not for our convention: Frame rotation

    R = R.transpose();
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

# Calculates Rotation Matrix given euler angles.
# Frame rotation matrix
def eulerAnglesToRotationMatrix(euler) :

    # euler = [phi; theta; psi] (roll, picth, yaw)
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(euler[0]), math.sin(euler[0])  ],
                    [0,        -math.sin(euler[0]), math.cos(euler[0])  ]
                    ])

    R_y = np.array([[math.cos(euler[1]),    0,     -math.sin(euler[1])  ],
                    [0,                     1,      0                   ],
                    [math.sin(euler[1]),    0,      math.cos(euler[1])  ]
                    ])

    R_z = np.array([[math.cos(euler[2]),    math.sin(euler[2]),     0],
                    [-math.sin(euler[2]),   math.cos(euler[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_x, np.dot( R_y, R_z ))

    return R

def quaternionsToRotationMatrix(q):
    # q = [qw, qx, qy, qz]

    R = np.array([[q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3],                     2*(q[1]*q[2] - q[0]*q[3]),                     2*(q[0]*q[2] + q[1]*q[3])],
                  [                    2*(q[1]*q[2] + q[0]*q[3]), q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3],                     2*(q[2]*q[3] - q[0]*q[1])],
                  [                    2*(q[1]*q[3] - q[0]*q[2]),                     2*(q[0]*q[1] + q[2]*q[3]), q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]]
                  ]);

    return R;

def rotationMatrixToQuaternion(R):
    # q = [qw, qx, qy, qz]

    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])/2;
    if qw == 0:
        print("Eroor: rotationMatrixToQuaternion qw = 0");
    qx = (R[2,1] - R[1,2])/float((4*qw));
    qy = (R[0,2] - R[2,0])/float((4*qw));
    qz = (R[1,0] - R[0,1])/float((4*qw));

    quat = [qw, qx, qy, qz];
    return quat;


if __name__ == '__main__':
    # Petit test:
    # latlon_orig_21 = np.array([32.654646, -85.315464 ]);
    # latlon_dest_21 = np.array([32.606556, -85.481728 ]);
    # print ('Lat/Lon orig:')
    # print(latlon_orig_21)
    # print ('Lat/Lon dest:')
    # print(latlon_dest_21)

    # ned = latlon_to_NED(latlon_orig_21, latlon_dest_21);
    # print ('NED: ')
    # print (ned);
    # latlon_des = NED_to_latlon(latlon_orig_21, ned);
    # print ('Lat/Lon dest computed:')
    # print(latlon_des)

    latlon_1_21 = np.array([32.606561,-85.482366]);
    latlon_2_21 = np.array([32.606395, -85.481866 ]);
    latlon_3_21 = np.array([32.606558, -85.481720 ]);
    latlon_4_21 = np.array([32.606403, -85.481687 ]);

    ned = latlon_to_NED(latlon_1_21, latlon_1_21);
    print ('NED: 1')
    print (ned);

    ned = latlon_to_NED(latlon_1_21, latlon_2_21);
    print ('NED: 2')
    print (ned);

    ned = latlon_to_NED(latlon_1_21, latlon_3_21);
    print ('NED: 3')
    print (ned);

    ned = latlon_to_NED(latlon_1_21, latlon_4_21);
    print ('NED: 4')
    print (ned);

    eulers_init = [np.deg2rad(60), np.deg2rad(54) , np.deg2rad(-20)];
    rot_CF_F = eulerAnglesToRotationMatrix(eulers_init);
    eulers_dest = rotationMatrixToEulerAngles(rot_CF_F);

    print('eulers_init:');
    print(eulers_init);
    print('eulers_dest:');
    print(eulers_dest);

    # Test Quaternion to Rotation
    eulers_init = [np.deg2rad(60), np.deg2rad(54) , np.deg2rad(-20)];
    rot_CF_F = eulerAnglesToRotationMatrix(eulers_init);
    q_CF_F = rotationMatrixToQuaternion(rot_CF_F);
    rot_CF_F_compu = quaternionsToRotationMatrix(q_CF_F);

    print('q_CF_F:')
    print(q_CF_F)
    print('rot_CF_F')
    print(rot_CF_F)
    print('rot_CF_F_compu')
    print(rot_CF_F_compu)


    phi_i = -np.pi +0.2;
    phi_f = -np.pi + 0.3;

    phi_diff = compute_angle_diff(phi_f, phi_i);
    print('compute_angle_diff: [Error] phi_f:{} phi_i:{} phi_diff:{}'.format(phi_f, phi_i, phi_diff));
