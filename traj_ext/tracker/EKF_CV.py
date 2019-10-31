
########################################################################################
#
# Implementation of an Extended Kalman Filter for Vehicle Tracking from Image Detection
# Child Class : Constant Velocity dynamic model
#
########################################################################################
from .EKF import *

class EKF_CV_track(EKF_track):

    def __init__(self, Q_mat, R_mat, P_init, track_id, label, x_init = None, t_current_ms = None):

        # Init parent
        EKF_track.__init__(self, 'CV', Q_mat, R_mat, P_init, track_id, label, x_init, t_current_ms);

        # As A is constant in this model, store it here for this model
        # Avoid re-constructing A each time we need to access to it
        self.A = np.array([[ 0,  0,  1,  0],\
                           [ 0,  0,  0,  1],\
                           [ 0,  0,  0,  0],\
                           [ 0,  0,  0,  0]]);

        # Psi defined as the angle of the velocity vector, in NED (Z down positive)
        self.psi_rad = 0;

        # Model: Point Mass Constant Velocity Model:
        # State: [x, y, v_x, v_y]
        # x_dot = v_x
        # y_dot = v_y
        # vx_dot = w_(vx_dot) - Noise
        # vy_dot = w_(vy_dot) - Noise

    @classmethod
    def get_default_param(cls):

        # Process noise
        Q = np.array([[ 0,  0,    0,    0],\
                       [ 0,  0,    0,    0],\
                       [ 0,  0,    0.1,    0],\
                       [ 0,  0,    0,    0.1]], np.float64);

        # Measurement noise: Pixel Noise
        R = np.array([[ 10,  0],\
                       [  0, 10]], np.float64);


        # Init Covariance
        P_init = np.array([[ 0.1,  0,    0,    0],\
                            [ 0,  0.1,    0,    0],\
                            [ 0,  0,    10,    0],\
                            [ 0,  0,    0,    10]], np.float64);


        return P_init, Q, R;

    def get_state_dim(self):

        # Dimension of the state:
        state_dim = 4;

        return state_dim;

    def create_x_init(self, psi_rad, x, y, vx, vy):

        # Create init state:
        x_i = np.array([x, y, vx, vy], np.float64);
        x_i.shape = (self.state_dim,1);

        return x_i;

    def get_A_model(self, x_current):
        return self.A;

    def compute_meas_H(self, cam_model, x_current = None):

        if x_current is None:
            x_current = self.x_current;

        # Use function from the camera model
        pos_F = np.asarray(x_current[0:2,0]).reshape(-1);
        pos_F = np.append(pos_F, 0);

        # Get H_pos = d([p_u;p_v])/d([pos_x,pos_y,pos_z])
        H_pos = cam_model.compute_meas_H(pos_F);

        # We want H = d([p_u;p_v])/d([pos_x,pos_y,v_x, v_y])
        H = np.zeros((2,4));
        H[:,0:2] = H_pos[0:2,0:2];

        return H;

    def propagate_state(self, state_current, delta_s):

        # Define discretize version of Process and Process noise matrix
        # First order approx
        F = np.identity(self.state_dim) + self.A*delta_s;

        # Propagate the current state
        state_current.shape = (self.state_dim,1);
        state_prop = F.dot(state_current);

        return state_prop;

    def trajpoint_from_state(self, state_data):

        traj_point = None;
        if not (state_data is None):

            # Extract Position, Velocity, Orientation
            x = state_data.x_state[0,0];
            y = state_data.x_state[1,0];

            vx = state_data.x_state[2,0];
            vy = state_data.x_state[3,0];

            psi_rad = None;

            if np.sqrt(vx*vx + vy*vy) > 0.5:

                if fabs(vx) < 0.05 :
                    pi_over_2 = float(math.pi)/float(2);
                    psi_rad =  math.copysign(pi_over_2, float(vy));

                else:
                    psi_rad = (np.arctan2(float(vy),float(vx)));

            traj_point = trajectory.TrajPoint(state_data.time_ms, x, y, vx, vy, psi_rad);

        return traj_point;

    def compute_processed_parameters(self, state_data):

        xy = vxy = psi_rad = None;
        if not (state_data is None):

            # Extract Position, Velocity, Orientation
            xy = np.array([state_data.x_state[0,0], state_data.x_state[1,0]])
            vxy = np.array([state_data.x_state[2,0], state_data.x_state[3,0]])

            if np.linalg.norm(vxy) > 0.5:

                if fabs(vxy[0]) < 0.05 :
                    pi_over_2 = float(math.pi)/float(2);
                    psi_rad =  math.copysign(pi_over_2, float(vxy[1]));

                else:
                    psi_rad = (np.arctan2(float(vxy[1]),float(vxy[0])));

        return xy, vxy, psi_rad;


    def get_processed_parameters_filter(self, current_time_ms):

        # Fetch filtered state: Fetch the fused or the filt one
        state_data = self.get_tk_fuse(current_time_ms);
        if (state_data is None):
            state_data = self.get_tk_predict(current_time_ms);

        xy = vxy = psi_rad = None;
        if not (state_data is None):

            xy, vxy, psi_rad = self.compute_processed_parameters(state_data);

        return xy, vxy, psi_rad;

    def get_processed_parameters_smoothed(self, current_time_ms):

        # Fetch smoothed state
        state_data = self.get_tk_smooth(current_time_ms);

        xy = vxy = psi_rad = None;
        if not (state_data is None):

            xy, vxy, psi_rad = self.compute_processed_parameters(state_data);

        return xy, vxy, psi_rad;