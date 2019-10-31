########################################################################################
#
# Implementation of an Extended Kalman Filter for Vehicle Tracking from Image Detection
# Child Class : Bicycle dynamic model
#
########################################################################################
import matplotlib
matplotlib.use('TkAgg')

from .EKF import *

lr = 1.5;
lf = 1.5;
l = 1.5;

class EKF_BM2_track(EKF_track):

    def __init__(self, Q_mat, R_mat, P_init, track_id, label, x_init = None, t_current_ms = None):

        # Init parent
        EKF_track.__init__(self, 'BM2',Q_mat, R_mat, P_init, track_id, label, x_init, t_current_ms);

        # Model: Bycicle model

        # State: [x, y, v, psi, beta]
        # x: position of cg
        # y: position of cg
        # v_r: velocity rear wheel
        # psi: orientation of the bicycle
        # delta: front wheel displacement (steerwheel angle)

    @classmethod
    def get_default_param(cls):

        # Process noise
        Q = np.array([[ 0,  0,     0,    0,     0],\
                       [ 0,  0,     0,    0,     0],\
                       [ 0,  0,     1,    0,     0],\
                       [ 0,  0,     0,      0,     0],\
                       [ 0,  0,     0,    0,     0.001]], np.float64);

        # Measurement noise: Pixel Noise
        R = np.array([[ 10,  0],\
                       [  0, 10]], np.float64);

        # Init Covariance
        P_init = np.array([[   0.01,     0,    0,    0,    0],\
                            [   0,     0.01,    0,    0,    0],\
                            [   0,     0,    0.02,    0,    0],\
                            [   0,     0,    0,    0.01,    0],\
                            [   0,     0,    0,    0,     0.001]], np.float64);

        return P_init, Q, R;

    def get_state_dim(self):

        # Dimension of the state:
        state_dim = 5

        return state_dim;

    def create_x_init(self, psi_rad, x, y, vx, vy):

        # Create init state:
        x_i = np.array([x, y, np.sqrt(vx*vx + vy*vy), psi_rad, 0.0], np.float64);
        x_i.shape = (self.state_dim,1);

        return x_i;

    def get_A_model(self, x_current):

        # Get values form state
        pos_x   = x_current[0,0];
        pos_y   = x_current[1,0];
        v       =  (x_current[2,0]);#abs(x_current[2,0]);
        psi     = self.clamp(x_current[3,0], -np.pi, np.pi);
        beta = self.clamp(x_current[4,0], -np.pi, np.pi);

        # Derivative of continuous model dynamic function: x_dot = f(x)
        # A = df/dx_state

        # psi_dot = float(v)/(lr+lf)*np.tan(theta);

        # Derivative of F function: x_(k+1)= f(x_k)
        A = np.array([[ 0,  0,                    np.cos(psi + beta),  -v*np.sin(psi+ beta),                              -v*np.sin(psi + beta)],\
                      [ 0,  0,                      np.sin(psi+beta),  v*np.cos(psi + beta),                               v*np.cos(psi + beta)],\
                      [ 0,  0,                                     0,                     0,                                                  0],\
                      [ 0,  0,              np.sin(beta)/(float(l)),                      0,                 (float(v)/float(l))*(np.cos(beta))],\
                      [ 0,  0,                                     0,                     0,                                                  0]]);

        return A;

    def compute_meas_H(self, cam_model, x_current = None):


        if x_current is None:
            x_current = self.x_current;

        # Use function from the camera model
        pos_F = np.asarray(x_current[0:2,0]).reshape(-1);

        pos_F = np.append(pos_F, 0);
        H_pos = cam_model.compute_meas_H(pos_F);

        # We want H = d([p_u;p_v])/d([x,y,v, psi, theta])
        H = np.zeros((2,5));
        H[:,0:2] = H_pos[0:2,0:2];

        return H;

    def propagate_state(self, x_current, delta_s):

        # Get values form state
        pos_x   = x_current[0,0];
        pos_y   = x_current[1,0];
        v       = x_current[2,0]; #abs(x_current[2,0]);
        psi     = self.clamp(x_current[3,0], -np.pi, np.pi);
        beta = self.clamp(x_current[4,0], -np.pi, np.pi);

        # Compute x_dot

        psi_dot = (float(v)/float(l))*np.sin(beta);
        x_dot = np.array([[v*np.cos(psi + beta),\
                           v*np.sin(psi + beta), \
                            0,\
                            psi_dot, \
                            0]]);
        x_dot.shape = (self.state_dim,1);

        # State predict
        x_current.shape = (5,1);
        x_prop = x_current + delta_s*x_dot;
        x_prop[3,0] = self.clamp(x_prop[3,0], -np.pi, np.pi);
        x_prop[4,0] = self.clamp(x_prop[4,0], -np.pi, np.pi);
        # x_prop[2,0] = abs(x_prop[2,0]);

        x_current.shape = (self.state_dim,1);

        return x_prop;

    def trajpoint_from_state(self, state_data):

        traj_point = None;
        if not (state_data is None):

            # Extract Position, Velocity, Orientation
            x = state_data.x_state[0,0];
            y = state_data.x_state[1,0];

            psi_rad = wraptopi(state_data.x_state[3,0]);

            vx = np.cos(psi_rad)*state_data.x_state[2,0];
            vy = np.sin(psi_rad)*state_data.x_state[2,0];

            traj_point = trajectory.TrajPoint(state_data.time_ms, x, y, vx, vy, psi_rad);

        return traj_point;


    def get_processed_parameters_filter(self, current_time_ms):

        # Fetch filtered state: Fetch the fused or the filt one
        state_data = self.get_tk_fuse(current_time_ms);
        if (state_data is None):
            state_data = self.get_tk_predict(current_time_ms);

        xy = vxy = psi_rad = None;
        if not (state_data is None):

            # Extract Position, Velocity, Orientation
            xy = np.array([state_data.x_state[0,0], state_data.x_state[1,0]])
            vxy = np.array([np.cos(state_data.x_state[3,0])*state_data.x_state[2,0],  np.sin(state_data.x_state[3,0])*state_data.x_state[2,0]])
            psi_rad = wraptopi(state_data.x_state[3,0])

        return xy, vxy, psi_rad;

    def get_processed_parameters_smoothed(self, current_time_ms):


        state_data = self.get_tk_smooth(current_time_ms);

        xy = vxy = psi_rad = None;
        if not (state_data is None):

            # Extract Position, Velocity, Orientation
            xy = np.array([state_data.x_state[0,0], state_data.x_state[1,0]])
            vxy = np.array([np.cos(state_data.x_state[3,0])*state_data.x_state[2,0],  np.sin(state_data.x_state[3,0])*state_data.x_state[2,0]])
            psi_rad = wraptopi(state_data.x_state[3,0])

        return xy, vxy, psi_rad;


    def clamp(self, n, smallest, largest):
        return n
        # return max(smallest, min(n, largest))
        # return self.wraptopi(n);

    def wraptopi(self, x):
        pi = np.pi
        x = x - np.floor(x/(2*pi)) *2 *pi
        if x >= pi:
            x = x- 2*pi
        return x