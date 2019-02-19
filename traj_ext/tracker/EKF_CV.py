########################################################################################
#
# Implementation of an Extended Kalman Filter for Vehicle Tracking from Image Detection
# Child Class : Constant Velocity dynamic model
#
########################################################################################
from .EKF import *

class EKF_CV_track(EKF_track):

    def __init__(self, Q_mat, R_mat, x_init, P_init, track_id, t_current_ms, label):

        # Init parent
        EKF_track.__init__(self, Q_mat, R_mat, x_init, P_init, track_id, t_current_ms, label);

        # As A is constant in this model, store it here for this model
        # Avoid re-constructing A each time we need to access to it
        self.A = np.array([[ 0,  0,  1,  0],\
                           [ 0,  0,  0,  1],\
                           [ 0,  0,  0,  0],\
                           [ 0,  0,  0,  0]]);

        # Phi defined as the angle of the velocity vector, in NED (Z down positive)
        self.phi_rad = 0;

        # Model: Point Mass Constant Velocity Model:
        # State: [x, y, v_x, v_y]
        # x_dot = v_x
        # y_dot = v_y
        # vx_dot = w_(vx_dot) - Noise
        # vy_dot = w_(vy_dot) - Noise

    def get_state_dim(self):

        # Dimension of the state:
        state_dim = 4;

        return state_dim;

    def create_x_init(self, box3D):

        # Get position from 3D box:
        pos_F = np.array(box3D[1:4]);
        pos_F.shape = (3,1)

        # Create init state:
        x_i = np.matrix([pos_F[0,0],pos_F[1,0], 0, 0], np.float64);
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

            traj_point = TrajPoint(state_data.time_ms, x, y, vx, vy, psi_rad);

        return traj_point;

    def compute_processed_parameters(self, state_data):

        xy = vxy = phi_rad = None;
        if not (state_data is None):

            # Extract Position, Velocity, Orientation
            xy = np.array([state_data.x_state[0,0], state_data.x_state[1,0]])
            vxy = np.array([state_data.x_state[2,0], state_data.x_state[3,0]])

            if np.linalg.norm(vxy) > 0.5:

                if fabs(vxy[0]) < 0.05 :
                    pi_over_2 = float(math.pi)/float(2);
                    phi_rad =  math.copysign(pi_over_2, float(vxy[1]));

                else:
                    phi_rad = (np.arctan2(float(vxy[1]),float(vxy[0])));

        return xy, vxy, phi_rad;


    def get_processed_parameters_filter(self, current_time_ms):

        # self.update_phi_rad()
        # xy = np.array([self.x_current[0,0], self.x_current[1,0]])
        # vxy = np.array([self.x_current[2,0], self.x_current[3,0]])
        # phi_rad = self.phi_rad
        # box3D_meas = None

        # # Get box3d_filt if measurement, otherwise None
        # for obj in self.history_fuse_states:
        #     if obj['timestamp'] == current_time_ms:
        #         box3D_meas = copy.copy(self.box3D_meas)

        # return xy, vxy, phi_rad, box3D_meas

        # Fetch filtered state: Fetch the fused or the filt one
        state_data = self.get_tk_fuse(current_time_ms);
        if (state_data is None):
            state_data = self.get_tk_predict(current_time_ms);

        xy = vxy = phi_rad = None;
        if not (state_data is None):

            xy, vxy, phi_rad = self.compute_processed_parameters(state_data);

        return xy, vxy, phi_rad;

    def get_processed_parameters_smoothed(self, current_time_ms):

        # Fetch smoothed state
        state_data = self.get_tk_smooth(current_time_ms);

        xy = vxy = phi_rad = None;
        if not (state_data is None):

            xy, vxy, phi_rad = self.compute_processed_parameters(state_data);

        return xy, vxy, phi_rad;
        # # Update phi by the past smoothed phi; or by values of velocities if low

        # # If first timestamp, get phi from processed data
        # if len(self.history_smooth_params) == 0 :

        #     for obj in self.history_processed_states:
        #         if obj['timestamp'] == current_time:
        #             phi_rad_sm = obj['phi_rad']

        # else:

        #     # get timestamp of past time closest to current time
        #     time=[]
        #     for index, obj in enumerate(self.history_smooth_params):
        #         time.append(obj['timestamp'])
        #     index = min(range(len(time)), key=lambda i: abs(time[i]-current_time))

        #     # Update phi with appropriate value
        #     phi_rad_sm = self.history_smooth_params[index]['phi_rad']

        # if sqrt(pow(self.x_smooth[3,0],2) + pow(self.x_smooth[2,0],2)) > 0.8:
        #     phi_rad_sm = (np.arctan(float(self.x_smooth[3,0])/float(self.x_smooth[2,0])));

        #     # Make sure it's the correct phi
        #     if float(self.x_smooth[2,0]) < 0:
        #         phi_rad_sm = phi_rad_sm + pi;


        # xy = np.array([self.x_smooth[0,0], self.x_smooth[1,0]])
        # vxy = np.array([self.x_smooth[2,0], self.x_smooth[3,0]])

        # return xy, vxy, phi_rad_sm


    # TO DO repalce all current times in function by self.time_f
    def set_phi_rad(self, phi_rad, current_time_ms):
        """Manually set the orientation phi of the traker
           Use at init when velocity is still 0 in this dynamical model

        Args:
            phi_rad (TYPE): Orientation (yaw) phi of the box
        """
        self.phi_rad = phi_rad;

	# # TO DO repalce all current times in function by self.time_f
 #    def update_phi_rad(self):
 #        """Updates the value of the steering angle when low velocities

 #        Args:
 #            current_time (TYPE): Description
 #        """
 #        # Define the orientation of the box according to the velocity, if velocity > 0.1
 #        if sqrt(pow(self.x_current[3,0],2) + pow(self.x_current[2,0],2)) > 0.8:
 #            self.phi_rad = (np.arctan(float(self.x_current[3,0])/float(self.x_current[2,0])));

 #            # Make sure it's the correct phi
 #            if float(self.x_current[2,0]) < 0:
 #                self.phi_rad = self.phi_rad + pi;

 #        self.phi_rad = wraptopi(self.phi_rad)