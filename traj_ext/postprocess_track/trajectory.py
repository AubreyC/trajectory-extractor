#################################################################################
#
# Trajectory structure for a vehicle
#
#################################################################################

import numpy as np
import os
import sys
import cv2
import collections
import pandas as pd

from traj_ext.postprocess_track import trajutil
from traj_ext.utils import mathutil
from traj_ext.utils import det_zone

from traj_ext.box3D_fitting import box3D_object

from traj_ext.postprocess_track.time_ignore import TimeIgnore

from traj_ext.object_det.det_object import DetObject
from traj_ext.utils import cfgutil


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
    def __init__(self, id, agent_type = '', length = 0., width = 0., height = 0.,  color = None):

        self._traj = [];

        self._list_box3D_type = box3D_object.Type3DBoxStruct.default_3DBox_list();

        self._length = length;
        self._width = width;
        self._height = height;

        self._id = id;
        self._agent_type = agent_type;
        self.set_agent_type(agent_type);

        self._color = (int(np.random.randint(0,255,1)[0]), int(np.random.randint(0,255,1)[0]), int(np.random.randint(0,255,1)[0]));

        if not (color is None):
            self._color = color;


        # By default: Not complete
        self.complete = False;

    @classmethod
    def write_trajectory_panda_csv(cls, folder_path, name_prefix, traj_list, list_times_ms):
        """Write trajectories into a single csv file

        Args:
            folder_path (TYPE): Description
            name_prefix (TYPE): Description
            traj_list (TYPE): Description
            list_times_ms (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            NameError: Description
        """

        # Write trajectories
        csv_name = name_prefix +'.csv';
        df_traj_path = os.path.join(folder_path, csv_name);

        dict_traj_pd = collections.OrderedDict.fromkeys(['track_id',\
                                                         'frame_id',\
                                                         'timestamp_ms',\
                                                         'agent_type',\
                                                         'x',\
                                                         'y',\
                                                         'vx',\
                                                         'vy',\
                                                         'psi_rad',\
                                                         'length',\
                                                         'width']);
                                                         # 'height']);
        dict_traj_pd['track_id'] = [];
        dict_traj_pd['frame_id'] = [];
        dict_traj_pd['timestamp_ms'] = [];
        dict_traj_pd['agent_type'] = [];
        dict_traj_pd['x'] = [];
        dict_traj_pd['y'] = [];
        dict_traj_pd['vx'] = [];
        dict_traj_pd['vy'] = [];
        dict_traj_pd['psi_rad'] = [];
        dict_traj_pd['length'] = [];
        dict_traj_pd['width'] = [];
        # dict_traj_pd['height'] = [];

        for frame_index, timestamp_ms in enumerate(list_times_ms):

                for traj in traj_list:

                    # Define name:
                    traj_point = traj.get_point_at_timestamp(timestamp_ms);
                    if not(traj_point is None):

                        if not (timestamp_ms == traj_point.time_ms):
                            raise NameError('[ERROR]: write_trajectory_csv (timestamp_ms != traj.time_ms)')

                        # Data:
                        dict_traj_pd['track_id'].append(traj.get_id());
                        dict_traj_pd['frame_id'].append(frame_index);

                        dict_traj_pd['timestamp_ms'].append(traj_point.time_ms);
                        dict_traj_pd['agent_type'].append(traj.get_agent_type());

                        l, w, h = traj.get_size();
                        dict_traj_pd['length'].append(l);
                        dict_traj_pd['width'].append(w);
                        # dict_traj_pd['height'].append(h);

                        dict_traj_pd['x'].append(traj_point.x);
                        dict_traj_pd['y'].append(traj_point.y);
                        dict_traj_pd['vx'].append(traj_point.vx);
                        dict_traj_pd['vy'].append(traj_point.vy);
                        dict_traj_pd['psi_rad'].append(traj_point.psi_rad);

        # Create dataframe
        df_traj = pd.DataFrame(dict_traj_pd);

        # Sort by track_id:
        df_traj.sort_values(by=['track_id', 'timestamp_ms'], inplace = True);

        # Write dataframe in csv
        df_traj.to_csv(df_traj_path, index=False);

        return;

    @classmethod
    def read_trajectory_panda_csv(cls, traj_panda_csv_path):
        """Read trajectories from a single trajectory csv file

        Args:
         traj_panda_csv_path (TYPE): Path to the csv file
        """

        # Read dataframe with panda
        df = pd.read_csv(traj_panda_csv_path);

        grouped = df.groupby(['track_id'], sort=False);

        traj_list = [None] * grouped.ngroups

        current_traj = 0;
        for track_id, rows in grouped:


            agent_type = rows['agent_type'].values[0];
            length = rows['length'].values[0];
            width = rows['width'].values[0];
            # height = rows['height'].values[0];

            traj = Trajectory(track_id, agent_type, length = length,  width = width);

            for index, row in rows.iterrows():
                time_ms = row['timestamp_ms'];
                x = row['x']
                y = row['y']
                vx = row['vx'];
                vy = row['vy'];
                psi_rad = row['psi_rad'];

                traj.add_point(time_ms, x, y, vx, vy, psi_rad);

            traj_list[current_traj] = traj;

            # Display status
            status_str = 'Reading traj: {}/{}'.format(current_traj, grouped.ngroups);
            cfgutil.progress_bar(current_traj, grouped.ngroups, status_str)

            current_traj += 1;

        print('');
        return traj_list;

    def set_agent_type(self, agent_type):


        found_type = False;
        for box3D_type in self._list_box3D_type:

            if  box3D_type.label == agent_type:
                self._length = box3D_type.length;
                self._width = box3D_type.width;
                self._height = box3D_type.height;
                self._agent_type = agent_type;

                found_type = True;

        if not found_type:
            raise ValueError('agent_type not recognized: {}'.format(agent_type));

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

    def get_distance_covered(self):
        """Calculate the total distance covered by the trajectory in meters

        Returns:
            TYPE: Total covered distance in meters
        """

        traj_point_previous = None;
        distance_m = 0;
        for traj_point in self._traj:

            if not (traj_point_previous is None):

                # Crude integration
                delta_m = np.linalg.norm(np.array([traj_point_previous.x - traj_point.x, traj_point_previous.y - traj_point.y]));

                # Sum up
                distance_m = distance_m + delta_m;

            traj_point_previous = traj_point;

        # print('[Trajectory]: id: {} distance: {} meters'.format(self.get_id(), distance_m));

        return distance_m;

    def check_startend_time_ignore(self, time_ignore_list, check_time_start = True, check_time_end = False):
        """Check if the trajectory is inside the time_ignore time frame

        Args:
            time_ignore_list (TYPE): List of time_ignore object
            check_time_start (bool, optional): Check if the start point is inside the time_ignore time frame
            check_time_end (bool, optional): Check if the end point is inside the time_ignore time frame

        Returns:
            TYPE: Return True if inside
        """
        time_inside = False;
        for time_ignore in time_ignore_list:

            time_start_inside = False;
            if check_time_start:
                traj_point = self.get_start_trajoint();
                time_start_inside = time_ignore.check_time_inside(traj_point.time_ms);

            time_end_inside = False;
            if check_time_end:
                traj_point = self.get_end_trajoint();
                time_end_inside = time_ignore.check_time_inside(traj_point.time_ms);

            time_inside = time_end_inside or time_start_inside;

            if time_inside:
                break;

        return time_inside;

    def remove_point_in_time_ignore(self, time_ignore_list, det_zone_FNED_list = []):

        for traj_point in list(self._traj):

            for time_ignore in time_ignore_list:

                if time_ignore.check_time_inside(traj_point.time_ms):
                    print('time_ignore')
                    if len(det_zone_FNED_list) > 0:

                        for det_zone_FNED in det_zone_FNED_list:
                            print('det_zone_FNED')

                            point_FNED = np.array([traj_point.x, traj_point.y], np.float);

                            if det_zone_FNED.in_zone(point_FNED):
                                print('2: {}'.format(traj_point.time_ms))
                                print('point_FNED: {}'.format(point_FNED))

                                self._traj.remove(traj_point);

                    else:

                        self._traj.remove(traj_point);



    def check_is_complete(self, det_zone_FNED):
        """Check if a trajectory is complete:

        Args:
            det_zone_FNED (TYPE): Detection zone used to check if complete

        Returns:
            TYPE: Boolean flag
        """
        complete = False;
        if self.get_length() > 1:

            trajoint_start = self.get_start_trajoint();
            trajoint_end = self.get_end_trajoint();

            # Check if it starts or end inside the complete detection zone
            start_in = det_zone_FNED.in_zone(np.array([[trajoint_start.x], [trajoint_start.y]]));
            end_in = det_zone_FNED.in_zone(np.array([[trajoint_end.x], [trajoint_end.y]]));

            complete = True;
            if start_in or end_in:
                complete = False;

        self.complete = complete;
        # print('Traj :" start_in:{} end_in:{}'.format(start_in, end_in));

        return complete;

    def delete_point_outside(self, det_zone_FNED):
        """Delete points that are outside the detection zone

        Args:
            det_zone_FNED (TYPE): Detection Zone

        Returns:
            TYPE: Description
        """
        if self.get_length() > 0:

            for traj_point in list(self._traj):

                if not det_zone_FNED.in_zone(np.array([[traj_point.x], [traj_point.y]])):

                    self._traj.remove(traj_point);

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

    def set_size(self, length, width, height):
        """Set the agent's size

        Args:
            width (TYPE): Width of the agent in meters
            length (TYPE): Length of the agent in meters
            height (TYPE): Height of the agent in meters

        """
        self._width = width;
        self._length = length;
        self._height = height;

    def get_size(self):
        """Return the agent's size

        Returns:
            TYPE: length, width, height in meters
        """
        return self._length, self._width, self._height;

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
            if time_ms >= self._traj[0].time_ms and time_ms <= self._traj[-1].time_ms:

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

    def get_start_trajoint(self):
        """Return the trajectory starting point

        Returns:
            TYPE: trajpoint
        """

        if len(self._traj) < 1:
            return None;

        return self._traj[0];


    def get_end_trajoint(self):
        """Return the trajectory end point

        Returns:
            TYPE: trajpoint
        """

        if len(self._traj) < 1:
            return None;

        return self._traj[-1];

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

    def complete_missing_psi_rad(self):
        """ Replace missing psi_rad (set to None) by the last "not None" previous psi_rad.
            Usefull when generating trajectory from EKF_CV where psi_rad is estimated from velocity.

        Raises:
            NameError: Description
        """

        # Find the first not None psi_rad:
        psi_rad_first = None;
        for traj_point in self._traj:

            if not (traj_point.psi_rad is None) :
                psi_rad_first = traj_point.psi_rad;
                break;

        # If psi_rad_first is still None, set it to 0
        if psi_rad_first is None:
            psi_rad_first = 0;

        # Replace the psi_rad that are None with the last not None psi_rad:
        psi_rad_last = psi_rad_first;
        for traj_point in self._traj:

            # Replace with last not None
            if traj_point.psi_rad is None:
                traj_point.psi_rad = psi_rad_last;

            # Keep last psi_rad
            psi_rad_last = traj_point.psi_rad;


            if traj_point.psi_rad is None:
                raise NameError('traj[index].psi_rad None index: {}'.format(index))


    def display_on_image(self, time_ms, image, cam_model, only_complete = False, color_text = (0, 0, 255), no_label = False, custom_text = None, complete_marker = False, velocity_label = False):
        """Display trajectory at time_ms on image

        Args:
            time_ms (TYPE): Time in ms
            image (TYPE): Image
            cam_model (TYPE): Camera Model
            color_text (tuple, optional): text color
            color (None, optional): Color
            no_label (bool, optional): Flag
            custom_text (None, optional): Custom text

        Deleted Parameters:
            image (TYPE): Image
        """


        # Make sure the image is not null
        det_track = None;
        if not (image is None):

            # In only_complete = True, then it has to be complete
            if (not only_complete) or self.complete:

                # Get trajectory point for specific time_ms
                traj_point = self.get_point_at_timestamp(time_ms);

                if traj_point:

                    # Create 3D Box
                    length, width, height = self.get_size();
                    box3D = box3D_object.Box3DObject(traj_point.psi_rad,\
                                                     traj_point.x,\
                                                     traj_point.y,\
                                                     0.0,\
                                                     length,\
                                                     width,\
                                                     height);

                    # Display 3D box on image
                    image = box3D.display_on_image(image, cam_model, color = self.get_color())

                    # Create a 2D box from the 3D box
                    det_2Dbox = box3D.get_projected_2Dbox(cam_model);
                    det_track = DetObject(self.get_id(), self._agent_type, det_2Dbox, 0.99, self.get_id());

                    # Reproject on the satellite image
                    pt_pix = cam_model.project_points(np.array([(traj_point.x, traj_point.y, 0.0)]));
                    pt_pix = (int(pt_pix[0]),int(pt_pix[1]));
                    image = cv2.circle(image, pt_pix,3, self.get_color(), -1);

                    # Add Velocity annotations to the track:
                    if velocity_label:
                        v = np.sqrt(traj_point.vx*traj_point.vx + traj_point.vy*traj_point.vy);
                        text = "id:%i v:%.2fm/s" % (self.get_id(), v);

                    else:
                        # Add ID annotations to the track:
                        text = "id:%i" % (self.get_id());

                    if not (custom_text is None):
                        text = custom_text;

                    if not no_label:
                        image = cv2.putText(image, text, pt_pix, cv2.FONT_HERSHEY_COMPLEX, 0.6, color_text, 1)

                    if complete_marker:
                        if not self.complete:
                            image = cv2.circle(image, pt_pix,20, (0,0,255), 3);


        return image, det_track;


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

    @classmethod
    def generate_metadata(cls, traj_list, list_times_ms):
        """Generate meta data for a list of trajectories:
        total_traj_nb: Total number of trajectory
        total_traj_time_s: Total time of the trajectories
        total_traj_distance_m: Total distance covered by the trajectories

        Args:
            traj_list (TYPE): List of trajectories

        Returns:
            TYPE: total_traj_nb, total_traj_time_s, total_traj_distance_m
        """

        # Write meta data
        duration_s = 0;
        if len(list_times_ms) > 2:
            duration_s = float(list_times_ms[-1] - list_times_ms[0])/float(1e3);

        total_traj_distance_m = 0;
        total_ms = 0;

        total_traj_nb = len(traj_list);

        for traj in traj_list:

            duration_ms = traj.get_length_ms();
            total_ms = total_ms + duration_ms;

            distance_m = traj.get_distance_covered();
            total_traj_distance_m = total_traj_distance_m + distance_m;


        total_traj_time_s = float(total_ms)/float(1e3);

        return total_traj_nb, total_traj_time_s, total_traj_distance_m, duration_s;
