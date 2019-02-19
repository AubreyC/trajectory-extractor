########################################################################################
#
# Implementation of an Extended Kalman Filter for Vehicle Tracking from Image Detection
# Child Class : Constant Velocity Constant Yaw Rate dynamic model
#
########################################################################################
import matplotlib
matplotlib.use('TkAgg')

from .EKF import *

class EKF_CVCYR_track(EKF_track):

    def __init__(self, Q_mat, R_mat, x_init, P_init, track_id, t_current_ms, label):

        # Init parent
        EKF_track.__init__(self, Q_mat, R_mat, x_init, P_init, track_id, t_current_ms, label);

        # Model: Unicyle model (Yaw and Velocity)

        # State: [x, y, v, phi, phi_dot]
        # x_dot = v*cos(phi)
        # y_dot = v*sin(phi)
        # v_dot = w_(v_dot) - Noise
        # phi_dot = phi_dot
        # phi_dot_dot = w_(phi_dot) - Noise


    def get_state_dim(self):

        # Dimension of the state:
        state_dim = 5

        return state_dim;

    def create_x_init(self, box3D):

        # Get position from 3D box:
        pos_F = np.array(box3D[1:4]);
        pos_F.shape = (3,1)

        # Create init state:
        x_i = np.matrix([pos_F[0,0],pos_F[1,0], 0.5, np.deg2rad(box3D[0]), 0], np.float64);
        x_i.shape = (self.state_dim,1);

        return x_i;


    def get_A_model(self, x_current):

        # Get values form state
        pos_x   = x_current[0,0];
        pos_y   = x_current[1,0];
        v       = x_current[2,0];
        phi     = wraptopi(x_current[3,0]);
        phi_dot = x_current[4,0];

        # Derivative of continuous model dynamic function: x_dot = f(x)
        # A = df/dx_state

        A = np.matrix([[ 0,  0,  np.cos(phi),  -v*np.sin(phi),      0],\
                       [ 0,  0,  np.sin(phi),   v*np.cos(phi),      0],\
                       [ 0,  0,            0,                0,     0],\
                       [ 0,  0,            0,                0,     1],\
                       [ 0,  0,            0,                0,     0]]);

        return A;

    def compute_meas_H(self, cam_model, x_current = None):


        if x_current is None:
            x_current = self.x_current;

        # Use function from the camera model
        pos_F = np.asarray(x_current[0:2,0]).reshape(-1);

        pos_F = np.append(pos_F, 0);
        H_pos = cam_model.compute_meas_H(pos_F);

        # We want H = d([p_u;p_v])/d([x,y,v, phi, phi_dot])
        H = np.zeros((2,5));
        H[:,0:2] = H_pos[0:2,0:2];

        return H;

    def trajpoint_from_state(self, state_data):

        traj_point = None;
        if not (state_data is None):
            # Extract Position, Velocity, Orientation
            x = state_data.x_state[0,0];
            y = state_data.x_state[1,0];

            psi_rad = state_data.x_state[3,0];

            vx = np.cos(psi_rad)*state_data.x_state[2,0];
            vy = np.sin(psi_rad)*state_data.x_state[2,0];

            traj_point = TrajPoint(state_data.time_ms, x, y, vx, vy, psi_rad);

        return traj_point;

    def propagate_state(self, x_current, delta_s):

        # Get values form state
        pos_x   = x_current[0,0];
        pos_y   = x_current[1,0];
        v       = x_current[2,0];
        phi     = wraptopi(x_current[3,0]);
        phi_dot = x_current[4,0];

        # # Do not change the orientation if not moving (v < 0.4)
        # if abs(x_current[2,0]) < 1:
        #     x_current[4,0] = 0;

        # Compute x_dot
        x_dot = np.matrix([[v*np.cos(phi),v*np.sin(phi), 0, phi_dot, 0]]);
        x_dot.shape = (self.state_dim,1);

        # Propagate state
        x_current.shape = (self.state_dim,1);
        x_prop = x_current + delta_s*x_dot;
        x_prop[3,0] = wraptopi(x_prop[3,0]);
        x_prop.shape = (self.state_dim,1);

        # Force positive velocity
        if v < 0:
            v = -v;
            phi += phi + np.pi

        return x_prop;


    def get_processed_parameters_filter(self, current_time_ms):

        # xy = np.array([self.x_current[0,0], self.x_current[1,0]])
        # vxy = np.array([np.cos(self.x_current[3,0])*self.x_current[2,0],  np.sin(self.x_current[3,0])*self.x_current[2,0]])
        # phi_rad = wraptopi(self.x_current[3,0])
        # box3D_meas = None

        # # Get box3d_filt if measurement, otherwise None
        # for obj in self.history_fuse_states:
        #     if obj['timestamp'] == current_time:
        #         box3D_meas = copy.copy(self.box3D_meas)

        # return  xy, vxy, phi_rad, box3D_meas

        # Fetch filtered state: Fetch the fused or the filt one
        state_data = self.get_tk_fuse(current_time_ms);
        if (state_data is None):
            state_data = self.get_tk_predict(current_time_ms);

        xy = vxy = phi_rad = None;
        if not (state_data is None):

            # Extract Position, Velocity, Orientation
            xy = np.array([state_data.x_state[0,0], state_data.x_state[1,0]])
            vxy = np.array([np.cos(state_data.x_state[3,0])*state_data.x_state[2,0],  np.sin(state_data.x_state[3,0])*state_data.x_state[2,0]])
            phi_rad = wraptopi(state_data.x_state[3,0])

        return xy, vxy, phi_rad;



    def set_phi_rad(self, phi_rad):
        """Manually set the orientation phi of the traker
           Use at init when velocity is still 0 in this dynamical model

        Args:
            phi_rad (TYPE): Orientation (yaw) phi of the box
        """
        self.phi_rad = phi_rad;


    def get_processed_parameters_smoothed(self, current_time_ms):

        # xy = np.array([self.x_smooth[0,0], self.x_smooth[1,0]])
        # vxy = np.array([np.cos(self.x_smooth[3,0])*self.x_smooth[2,0],  np.sin(self.x_smooth[3,0])*self.x_smooth[2,0]])
        # phi_rad = wraptopi(self.x_smooth[3,0])

        # return  xy, vxy, phi_rad
        # Fetch smoothed state

        state_data = self.get_tk_smooth(current_time_ms);

        xy = vxy = phi_rad = None;
        if not (state_data is None):

            # Extract Position, Velocity, Orientation
            xy = np.array([state_data.x_state[0,0], state_data.x_state[1,0]])
            vxy = np.array([np.cos(state_data.x_state[3,0])*state_data.x_state[2,0],  np.sin(state_data.x_state[3,0])*state_data.x_state[2,0]])
            phi_rad = wraptopi(state_data.x_state[3,0])

        return xy, vxy, phi_rad;

