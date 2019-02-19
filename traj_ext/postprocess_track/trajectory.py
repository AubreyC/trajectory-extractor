#################################################################################
#
# Trajectory structure for a vehicle
#
#################################################################################

import numpy as np
import os
import sys

from postprocess_track import trajutil
from utils import mathutil

class TrajPoint():
    def __init__(self, time_ms, x, y, vx, vy, psi_rad):
        self.time_ms = time_ms;
        self.x = x;
        self.y = y;
        self.vx = vx;
        self.vy = vy;
        self.psi_rad = psi_rad;

class Trajectory(object):

    """docstring for Trajectory"""
    def __init__(self, id, agent_type = '', color = None):

        self._traj = [];
        self._id = id;
        self._agent_type = agent_type;
        self._color = (int(np.random.randint(0,255,1)[0]), int(np.random.randint(0,255,1)[0]), int(np.random.randint(0,255,1)[0]));

        if not (color is None):
            self._color = color;

    def add_point(self, time_ms, x, y, vx, vy, psi_rad):
        """Add a point ot the trajectory

        Args:
            time_ms (TYPE): TimeStamp - milli-seconds
            x (TYPE): Position X of the center of the vehicle on the ground - meters
            y (TYPE): Position Y of the center of the vehicle on the ground - meters
            vx (TYPE): Velocity X of the center of the vehicle on the ground - meters/s
            vy (TYPE): Velocity Y of the center of the vehicle on the ground - meters/s
            psi_rad (TYPE): Orientation psi of the center of the vehicle on the ground - radians
        """
        data = TrajPoint(time_ms, x, y, vx, vy, psi_rad);

        self._traj.append(data);

    def get_id(self):
        """Return the trajectory id

        Returns:
            TYPE: Description
        """

        return self._id;

    def get_color(self):
        """Return the trajectory color

        Returns:
            TYPE: tulpe
        """

        return self._color;

    def get_time_ms_list(self):

        time_ms_list = [];
        for traj_point in self._traj:
            time_ms_list.append(traj_point.time_ms);

        return time_ms_list;

    def get_agent_type(self):
        """Return the agent's type

        Returns:
            TYPE: String
        """

        return self._agent_type;

    def get_point_at_timestamp(self, time_ms):
        """Summary

        Args:
            time_ms (TYPE): TimeStamp of the trajectory

        Returns:
            TYPE: Description
        """
        result = None;

        # Making sure traj is not empty
        if self.get_length() > 0:

            # First check if timestamp is withing the time frame of the trajetcory
            if time_ms > self._traj[0].time_ms and time_ms < self._traj[-1].time_ms:

                # Look for specific point
                for data in self._traj:
                    if data.time_ms == time_ms:
                        result = data;
                        break;

        return result;

    def get_length(self):
        """Return the lenght of the trajetcory in number of point

        Returns:
            TYPE: Description
        """
        return len(self._traj)

    def get_length_ms(self):
        """Return the lenght in milli-seconds of the trajectory

        Returns:
            TYPE: Length in of the trajectory - seconds
        """
        # If list empty or single element:
        if len(self._traj) < 2:
            return 0;

        else:
            time_length = self._traj[-1].time_ms - self._traj[0].time_ms;

            if time_length < 0:
                print('WARNING: Trajectory: time length is negative')
                time_length = - time_length;

            return time_length;

    def get_traj(self):
        """Return the trajectory list

        Returns:
            TYPE: trajectory_list
        """
        return self._traj;

    def get_index_for_time(self, time_ms):
        """Get the index of a specific timestamp in this trajectory

        Args:
            time_ms (TYPE): Timestamp in ms

        Returns:
            TYPE: Index of the corresponding timestamp in the trajectory
        """
        index = None;
        for i in range(len(self._traj)):
            if self._traj[i].time_ms == time_ms:
                index = i;
                break;

        return index;

    def compute_time_traj_overlap(self, traj_ext):

        start_t, end_t = trajutil.compute_time_traj_overlap(self, traj_ext);

        return start_t, end_t;


    def compute_error_to_traj(self, traj_ext, traj_ext_included = False):
        """Compute the distance point-to-points between two trajetcories

        Args:
            traj_ext (TYPE): Trajetcory to compute the distance to

        Returns:
            TYPE: Error vector with position error at each timestamp
        """

        # Get the start en end time to compare the two trajectories
        start_t, end_t = self.compute_time_traj_overlap(traj_ext);

        # In the case traj is included in traj_ext:
        if traj_ext_included:
            if start_t > self._traj[0].time_ms or end_t < self._traj[-1].time_ms:
                return None;

        # If no overlap between the two trajecories
        if(start_t > end_t):
            return None;

        # Get the index of the start and end:
        index_start_self = self.get_index_for_time(start_t);
        index_end_self = self.get_index_for_time(end_t);
        index_start_ext = traj_ext.get_index_for_time(start_t);
        index_end_ext = traj_ext.get_index_for_time(end_t);

        # Crop the trajectories for these start & end time
        traj_self = self._traj[index_start_self:index_end_self];
        traj_ext_dict = traj_ext.get_traj();
        traj_ext_dict = traj_ext_dict[index_start_ext:index_end_ext];

        # Make sure trajectories have the same size
        if not len(traj_self) == len(traj_ext_dict):
            print('[ERROR]: compute_distance_to_traj trajectories do not have the same size: {} {}'.format(len(traj_self), len(traj_ext_dict)));
            return None;

        # Create numpy array for X and Y position in time
        x_array_self = np.ones(len(traj_self), dtype='double');
        y_array_self = np.ones(len(traj_self), dtype='double');

        x_array_ext = np.ones(len(traj_ext_dict), dtype='double');
        y_array_ext = np.ones(len(traj_ext_dict), dtype='double');

        # Create numpy array for VX and VY position in time
        vx_array_self = np.ones(len(traj_self), dtype='double');
        vy_array_self = np.ones(len(traj_self), dtype='double');

        vx_array_ext = np.ones(len(traj_ext_dict), dtype='double');
        vy_array_ext = np.ones(len(traj_ext_dict), dtype='double');

        # Create numpy array for VX and VY position in time
        psi_array_self = np.ones(len(traj_self), dtype='double');

        psi_array_ext = np.ones(len(traj_ext_dict), dtype='double');


        for index in range(len(traj_self)):

            # Error for x and y
            x_array_self[index] = traj_self[index].x;
            y_array_self[index] = traj_self[index].y;

            x_array_ext[index] = traj_ext_dict[index].x;
            y_array_ext[index] = traj_ext_dict[index].y;

            # Error for vx and vy
            vx_array_self[index] = traj_self[index].vx;
            vy_array_self[index] = traj_self[index].vy;

            vx_array_ext[index] = traj_ext_dict[index].vx;
            vy_array_ext[index] = traj_ext_dict[index].vy;

            # Error for psi
            psi_array_self[index] = traj_self[index].psi_rad;
            psi_array_ext[index] = traj_ext_dict[index].psi_rad;

        # Compute distance betwen the points:
        x_error_sq = np.square(x_array_self - x_array_ext);
        y_error_sq = np.square(y_array_self - y_array_ext);
        error_xy = np.sqrt(x_error_sq + y_error_sq);

        # Compute distance betwen the points:
        vx_error_sq = np.square(vx_array_self - vx_array_ext);
        vy_error_sq = np.square(vy_array_self - vy_array_ext);
        error_vxvy = np.sqrt(vx_error_sq + vy_error_sq);

        psi_rad_error_sq = np.ones(len(traj_ext_dict), dtype='double');
        for index in range(len(traj_ext_dict)):
            psi_self = psi_array_self[index];
            psi_ext = psi_array_ext[index];

            if(abs(mathutil.compute_angle_diff(psi_self, psi_ext)) >2.5):
               psi_self = psi_self + np.pi;

            psi_rad_error_sq[index] = np.square( mathutil.compute_angle_diff(psi_self, psi_ext));

        error_psi_rad = np.sqrt(psi_rad_error_sq);

        return error_xy, error_vxvy, error_psi_rad;

    def compute_distance_to_traj(self, traj_ext, traj_ext_included = False):
        """Compute the distance point-to-points between two trajetcories

        Args:
            traj_ext (TYPE): Trajetcory to compute the distance to

        Returns:
            TYPE: Error vector with position error at each timestamp
        """

        # Get the start en end time to compare the two trajectories
        start_t, end_t = self.compute_time_traj_overlap(traj_ext);

        # print('start_t: {} end_t: {}'.format(traj_ext.get_traj()[0].time_ms, traj_ext.get_traj()[-1].time_ms));
        # print('start_t: {} end_t: {}'.format(self._traj[0].time_ms, self._traj[-1].time_ms));
        # print('start_t: {} end_t: {}'.format(start_t, end_t));
        # print('')

        # In the case traj is included in traj_ext:
        if traj_ext_included:
            if start_t > self._traj[0].time_ms or end_t < self._traj[-1].time_ms:
                return None;

        # If no overlap between the two trajecories
        if(start_t > end_t):
            return None;

        # Get the index of the start and end:
        index_start_self = self.get_index_for_time(start_t);
        index_end_self = self.get_index_for_time(end_t);
        index_start_ext = traj_ext.get_index_for_time(start_t);
        index_end_ext = traj_ext.get_index_for_time(end_t);

        # Crop the trajectories for these start & end time
        traj_self = self._traj[index_start_self:index_end_self];
        traj_ext_dict = traj_ext.get_traj();
        traj_ext_dict = traj_ext_dict[index_start_ext:index_end_ext];

        # Make sure trajectories have the same size
        if not len(traj_self) == len(traj_ext_dict):
            # print('start_t: {} end_t: {}'.format(traj_self[0].time_ms, traj_self[-1].time_ms));
            # print('start_t: {} end_t: {}'.format(traj_ext_dict[0].time_ms, traj_ext_dict[-1].time_ms));
            # print('start_t: {} end_t: {}'.format(start_t, end_t));
            print('[ERROR]: compute_distance_to_traj trajectories do not have the same size: {} {}'.format(len(traj_self), len(traj_ext_dict)));
            return None;

        # Create numpy array for X and Y position in time
        x_array_self = np.ones(len(traj_self), dtype='double');
        y_array_self = np.ones(len(traj_self), dtype='double');

        x_array_ext = np.ones(len(traj_ext_dict), dtype='double');
        y_array_ext = np.ones(len(traj_ext_dict), dtype='double');


        for index in range(len(traj_self)):

            x_array_self[index] = traj_self[index].x;
            y_array_self[index] = traj_self[index].y;

            x_array_ext[index] = traj_ext_dict[index].x;
            y_array_ext[index] = traj_ext_dict[index].y;

            # print('time_ms: {}'.format(traj_self[index].time_ms));
            # print('x: {} x_truth:{}'.format(traj_self[index].x , traj_ext_dict[index].x));
            # print('y: {} y_truth:{}'.format(traj_self[index].y , traj_ext_dict[index].y));

        # Compute distance betwen the points:
        x_error_sq = np.square(x_array_self - x_array_ext);
        y_error_sq = np.square(y_array_self - y_array_ext);

        # Compute distance error for each point
        error = np.sqrt(x_error_sq + y_error_sq);

        return error;