import os
from pickle import NONE
import sys
import pickle
from scipy.sparse import coo_matrix

import yaml

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

import copy
import numpy as np

from fairmotion.utils import utils,constants
from fairmotion.ops import conversions, math
from fairmotion.ops import quaternion
from misc.interaction import Interaction
import motion_utils
import time as tt
from envs import env_humanoid_base

class Env(env_humanoid_base.Env):
    def __init__(self, config):
        super().__init__(config)
        
        self._initialized = False
        self._config = copy.deepcopy(config)
        self._ref_motion = self._base_motion
        self._sensor_lookahead = self._config['state'].get('sensor_lookahead', [0.05, 0.15])
        self._start_time = np.zeros(self._num_agent)
        self._repeat_ref_motion = self._config.get('repeat_ref_motion', False)
        self._interaction_choices = self._config.get('interaction',['self'])
        self._compute_interaction_connectivity = self._config.get('interaction_connectivity',False)
        self._load_ref_interaction = self._config.get('load_ref_interaction',None)
        self._vr_sensors_link_idx = None
        if self._config['state'].get('vr_sensors_attached'):
            vr_link_names = self._config['state'].get('vr_sensors_attached')
            joint_idx = self._sim_agent[0]._char_info.joint_idx
            self._vr_sensors_link_idx = [joint_idx[n] for n in vr_link_names]

        if config.get('lazy_creation'):
            if self._verbose:
                print('The environment was created in a lazy fashion.')
                print('The function \"create\" should be called before it')
            return
        self.current_interaction = None
        self.create()

    def create(self):
        project_dir = self._config['project_dir']
        ref_motion_db = self._config['character'].get('ref_motion_db')
        ref_motion_file = motion_utils.collect_motion_files(project_dir, ref_motion_db)
        
        ''' Load Reference Motion '''

        self._ref_motion_all = []
        self._ref_motion_file_names = []
        self._sim_interaction_points = []
        self._kin_interaction_points = []
        for i in range(self._num_agent):
            ref_motion_all, ref_motion_file_names,ref_motion_file_asf_names= \
                motion_utils.load_motions(
                    ref_motion_file[i], 
                    None,
                    self._sim_agent[i]._char_info,
                    self._verbose)
            self._ref_motion_all.append(ref_motion_all)
            self._ref_motion_file_names.append(ref_motion_file_names)
            self._sim_interaction_points.append( None)
            self._kin_interaction_points.append( None)
       
        """Load Saved Refernce Interaction Connectivity"""
        if self._load_ref_interaction is not None:
            ref_interaction_path = os.path.join(project_dir, self._load_ref_interaction)
            self._load_ref_interaction = pickle.load(open(ref_interaction_path,'rb'))
           
        ''' Load Probability of Motion Trajectory '''
        prob_trajectory = self._config['reward'].get('prob_trajectory')
        if prob_trajectory:
            # TODO: Load Model
            self._prob_trajectory = None

        ''' Should call reset after all setups are done '''

        self.reset({'add_noise': False})

        self._initialized = True

        if self._verbose:
            print('----- Humanoid Imitation Environment Created -----')
            for i in range(self._num_agent):
                print('[Agent%d]: state(%d) and action(%d)' \
                      %(i, self.dim_state(i), self.dim_action(i)))
            print('-------------------------------')

    def callback_reset_prev(self, info):

        ''' Choose a reference motion randomly whenever reset '''
        
        self._ref_motion, self._ref_motion_idx = \
            self.sample_ref_motion(info.get('ref_motion_id'))
        
        ''' Choose a start time for the current reference motion '''
        
        if 'start_time' in info.keys():
            self._start_time = np.array(info.get('start_time'))
            assert self._start_time.shape[0] == self._num_agent
        else:
            for i in range(self._num_agent):
                self._start_time[i] = \
                    np.random.uniform(0.0, self._ref_motion[i].length())
        

    def callback_reset_after(self, info):
        for i in range(self._num_agent):
            self._kin_agent[i].set_pose(
                self._init_poses[i], self._init_vels[i])
            """Recompute cartesian states of all interaction points"""
            self._sim_interaction_points[i] = self.get_all_interaction_points(self._sim_agent[i],i)
            self._kin_interaction_points[i] = self.get_all_interaction_points(self._kin_agent[i],i)

            """Compute/Load interaction connectivity"""
            if self._compute_interaction_connectivity:
                self.current_interaction = {}
                edge_indices = self.compute_reference_interaction_mesh(self._kin_agent[i],i)
                self.current_interaction[i] = edge_indices

    def callback_step_after(self, actions, infos):
        ''' This is necessary to compute the reward correctly '''
        time = self.get_ref_motion_time()
        for i in range(self._num_agent):
            self._kin_agent[i].set_pose(
                self._ref_motion[i].get_pose_by_time(time[i]),
                self._ref_motion[i].get_velocity_by_time(time[i]))

            """Recompute cartesian states of all interaction points"""
            self._sim_interaction_points[i] = self.get_all_interaction_points(self._sim_agent[i],i)
            self._kin_interaction_points[i] = self.get_all_interaction_points(self._kin_agent[i],i)

            """Compute/Load interaction connectivity"""
            if self._compute_interaction_connectivity:
                self.current_interaction = {}
                edge_indices = self.compute_reference_interaction_mesh(self._kin_agent[i],i)
                self.current_interaction[i] = edge_indices

    def print_log_in_step(self):
        if self._verbose and self._end_of_episode:
            print('=================EOE=================')
            print('Reason:', self._end_of_episode_reason)
            print('TIME: (start:%02f) (elapsed:%02f) (time_after_eoe: %02f)'\
                %(self._start_time,
                  self.get_elapsed_time(),
                  self._time_elapsed_after_end_of_episode))
            print('=====================================')
    def compute_reference_interaction_mesh(self,agent, idx):
        if self._load_ref_interaction:
            motion_idx = self._ref_motion_idx[idx]
            current_time = self.get_ref_motion_time()
            current_frame = self._ref_motion[idx].time_to_frame(current_time)
            edge_indices = self._load_ref_interaction[motion_idx][current_frame][idx]
            return edge_indices
        else:
            all_points = []
            interaction_points = self.get_all_interaction_points(agent,idx)

            all_points.append(interaction_points[:,:3])
            all_points = np.concatenate(all_points,axis=0)
            interaction = Interaction(all_points)
            edge_indices = interaction.build_interaction_graph()
            return edge_indices
    def get_all_interaction_points(self,agent,idx,*args):
        interaction_points = []
        if "self" in self._interaction_choices:
            joint_candidate = np.array(agent._char_info.interaction_joint_candidate)
            joint_state = np.array(agent.get_joint_cartesian_state())
            interaction_points.append(joint_state[joint_candidate+1])
        if "ground" in self._interaction_choices:
            T = agent.get_facing_transform(self.get_ground_height(idx))
            _, _, v, _ = agent.get_root_state()
            R,p = conversions.T2Rp(T)
            v_facing = v - math.projectionOnVector(v, agent._char_info.v_up_env)

            ground_point = np.zeros((1,6))
            ground_point[0,:3] = p
            ground_point[0,3:6] = v_facing

            interaction_points.append(ground_point)
        return np.concatenate(interaction_points,axis=0)
    def compute_interaction_mesh(self,points1,points2,T=constants.EYE_T):

        R,p = conversions.T2Rp(T)
        # interaction_candidate1 = agent1._char_info.interaction_joint_candidate
        # interaction_candidate2 = agent2._char_info.interaction_joint_candidate

        # joint_state1 = agent1.get_joint_cartesian_state()
        # joint_state2 = agent2.get_joint_cartesian_state()


        # '''Interaction mesh Position'''
        # for i in interaction_candidate1:
        #     for j in interaction_candidate2:
        #         pi,vi = joint_state1[i+1][:3],joint_state1[i+1][3:]
        #         pj,vj = joint_state2[j+1][:3],joint_state2[j+1][3:]

        #         dp = R @ ((pj - pi) - p)
        #         dv = R @ (vj - vi)
        #         mesh_data[i+1,j+1] = np.hstack([dp,dv])

        # end_bf = time.time()
        # print ("Brute Force Time elapsed:", end_bf - start_bf)

        # joint_state1_np = np.array(joint_state1)
        # joint_state2_np = np.array(joint_state2)
        # interaction_candidate1_np = np.array(interaction_candidate1)
        # interaction_candidate2_np = np.array(interaction_candidate2)

        # int_i = joint_state1_np[interaction_candidate1_np+1]
        # int_j = joint_state2_np[interaction_candidate2_np+1]

        int_i = np.array(points1)
        int_j = np.array(points2)

        int_i_exp = np.expand_dims(int_i,1)
        int_i_repeat = np.repeat(int_i_exp,len(points2),1)
        int_j_exp = np.expand_dims(int_j,0)
        int_j_repeat = np.repeat(int_j_exp,len(points1),0)


        d_int = (int_j_repeat - int_i_repeat)
        d_int [:,:,:3] = d_int [:,:,:3] - p
        d_int  [:,:,:3]= d_int[:,:,:3] @ R.T
        d_int  [:,:,3:]= d_int[:,:,3:] @ R.T
        return d_int


    def compute_interaction_mesh_lower_triangle(self,agent,T=constants.EYE_T):
        if self._verbose:
            print("======Compute Interaction Mesh==========")
        R,p = conversions.T2Rp(T)
        interaction_candidate = agent._char_info.interaction_joint_candidate
        
        mesh_data = []
    
        joint_state = agent.get_joint_cartesian_state()

        '''Interaction mesh Position'''
        for i in interaction_candidate:
            for j in interaction_candidate:
                if j>=i :
                    continue
                pi,vi = joint_state[i+1][:3],joint_state[i+1][3:]
                pj,vj = joint_state[j+1][:3],joint_state[j+1][3:]

                dp = R @ ((pj - pi) - p)
                dv = R @ (vj - vi)
                mesh_data.append(np.hstack([dp,dv]))
        return mesh_data
    def compute_init_pose_vel(self, info):
        ''' This performs reference-state-initialization (RSI) '''
        init_poses, init_vels = [], []

        for i in range(self._num_agent):
            ''' Set the state of simulated agent by using the state of reference motion '''
            init_pose = self._ref_motion[i].get_pose_by_time(self._start_time[i])
            init_vel = self._ref_motion[i].get_velocity_by_time(self._start_time[i])
            ''' Add noise to the state if necessary '''
            if info.get('add_noise'):
                init_pose, init_vel = \
                    self._base_env.add_noise_to_pose_vel(
                        self._sim_agent[i], init_pose, init_vel)
            init_poses.append(init_pose)
            init_vels.append(init_vel)
        return init_poses, init_vels
    def get_general_interaction_graph(self,idx,agent_type="sim"):
        if agent_type == "sim":
            agent = self._sim_agent[idx]
            interaction_point_states = self._sim_interaction_points[idx]
        elif agent_type == "kin":
            agent = self._kin_agent[idx]
            interaction_point_states = self._kin_interaction_points[idx]
        else:
            raise NotImplementedError

        edge_indices = self.current_interaction
        row = edge_indices[0]
        col = edge_indices[1]
        data = np.ones_like(row)
        edge_indices_mtx = coo_matrix((data,(row,col)))
        edge_indices_mtx = edge_indices_mtx.todense()
        return np.array([])
    def get_state_by_key(self, idx, key):
        state = []

        ref_motion = self._ref_motion[idx] 
        time = self.get_ref_motion_time()
        if key=='body':
            start_time = tt.time()

            state.append(self.state_body(idx, "sim"))

        elif key=='ref_motion_abs' or key=='ref_motion_rel' or key=='ref_motion_abs_rel':
            start_time = tt.time()
            ref_motion_abs = True \
                if (key=='ref_motion_abs' or key=='ref_motion_abs_rel') else False
            ref_motion_rel = True \
                if (key=='ref_motion_rel' or key=='ref_motion_abs_rel') else False
            poses, vels = [], []
            for dt in self._sensor_lookahead:
                t = np.clip(
                    time[idx] + dt, 
                    0.0, 
                    ref_motion.length())
                poses.append(ref_motion.get_pose_by_time(t))
                vels.append(ref_motion.get_velocity_by_time(t))
            state.append(self.state_imitation(idx,
                                              poses,
                                              vels,
                                              include_abs=ref_motion_abs,
                                              include_rel=ref_motion_rel))
        elif key=="sim_interaction_graph_full":
            sim_agent = self._sim_agent[idx]
            sim_interaction_points = self._sim_interaction_points[idx]
            sim_ig = self.compute_interaction_mesh(sim_interaction_points,sim_interaction_points,T = sim_agent.get_facing_transform(self.get_ground_height(idx)))
            
            state.append(sim_ig.flatten())
        elif key == 'sim_interaction_points_height':
            sim_agent = self._sim_agent[idx]
            inter_point_height = self._sim_interaction_points[idx][:,:3] * sim_agent._char_info.v_up_env
            inter_point_height = np.sum(inter_point_height,axis=1)
            state.append(np.hstack(inter_point_height))
        elif key == "ref_interaction_points_height":
            sim_agent = self._sim_agent[idx]
            poses, vels = [], []
            for dt in self._sensor_lookahead:
                t = np.clip(
                    time[idx] + dt, 
                    0.0, 
                    ref_motion.length())
                poses.append(ref_motion.get_pose_by_time(t))
                vels.append(ref_motion.get_velocity_by_time(t))
            kin_joint_states = self.state_imitation_joint_cartesian_state(
                idx,
                poses,
                vels,
                only_height=True
            )
            state.append(kin_joint_states) 

        elif key=="ref_interaction_graph_full_abs":
            sim_agent = self._sim_agent[idx]
            poses, vels = [], []
            for dt in self._sensor_lookahead:
                t = np.clip(
                    time[idx] + dt, 
                    0.0, 
                    ref_motion.length())
                poses.append(ref_motion.get_pose_by_time(t))
                vels.append(ref_motion.get_velocity_by_time(t))
            kin_ig = self.state_imitation_interaction_mesh(
                idx,
                poses,
                vels,
                full=True)

            state.append((kin_ig))                  
        else:
            raise NotImplementedError

        return np.hstack(state)
    def dim_interaction(self,type="joint"):
        if type=="full":
            return self.compute_interaction_mesh(self._sim_interaction_points[0],self._sim_interaction_points[0]).flatten().shape[0]
        else:
            return np.array(self._sim_interaction_points[0]).flatten().shape[0]
    def num_interaction(self):
        sc = self._state_choices.copy()
        num = 0
        for i in sc:
            if "sim_interaction_graph_full" == i or "sim_interaction_points_state" == i:
                num+=1
            elif "ref_interaction_graph_full" in i or "ref_interaction_points_state" in i:
                num += len(self._sensor_lookahead)
        return num
    def state_imitation_joint_cartesian_state(self,idx,poses,vels,only_height = False):
        assert len(poses) == len(vels)
        kin_agent = self._kin_agent[idx]

        state = []
        state_kin_orig = kin_agent.save_states()
        for pose, vel in zip(poses, vels):
            kin_agent.set_pose(pose, vel)
            inter_points = self.get_all_interaction_points(kin_agent,idx)
            if only_height:
                inter_points = inter_points[:,:3] * kin_agent._char_info.v_up_env
                inter_points = np.sum(inter_points,axis=1)
            state.append(np.hstack(inter_points))
            
        kin_agent.restore_states(state_kin_orig)

        return np.hstack(state)
    def state_imitation_interaction_mesh(self,idx,poses,vels,full=False):
        assert len(poses) == len(vels)
        kin_agent = self._kin_agent[idx]
        
        state = []
        state_kin_orig = kin_agent.save_states()
        for pose, vel in zip(poses, vels):
            kin_agent.set_pose(pose, vel)
            self._kin_interaction_points[idx] = self.get_all_interaction_points(kin_agent,idx)
            kin_interaction_points = self._kin_interaction_points[idx]
            if not full:
                kin_ig = self.compute_interaction_mesh_lower_triangle(kin_agent,T = kin_agent.get_facing_transform(self.get_ground_height(idx)))
                state.append(np.hstack(kin_ig))
            else:
                kin_ig = self.compute_interaction_mesh(kin_interaction_points,kin_interaction_points,T = kin_agent.get_facing_transform(self.get_ground_height(idx)))
                state.append(kin_ig.flatten())

        kin_agent.restore_states(state_kin_orig)
        self._kin_interaction_points[idx] = self.get_all_interaction_points(kin_agent,idx)
        return np.hstack(state)
    
    def state_body(
        self, 
        idx, 
        agent="sim", 
        type=None, 
        return_stacked=True):
        if agent == "sim":
            agent = self._sim_agent[idx]
        elif agent == "kin":
            agent = self._kin_agent[idx]
        else:
            raise NotImplementedError
        return self._state_body(idx, agent, type, return_stacked)

    def state_task(self, idx):
        sc = self._state_choices.copy()
        if 'sim_interaction_graph' in sc:
            sc.remove('sim_interaction_graph')
        if 'body' in sc:
            sc.remove('body')
        return self.state(idx, sc)

    def state_imitation(
        self, 
        idx,
        poses, 
        vels, 
        include_abs, 
        include_rel):

        assert len(poses) == len(vels)

        sim_agent = self._sim_agent[idx]
        kin_agent = self._kin_agent[idx]

        R_sim, p_sim = conversions.T2Rp(
            sim_agent.get_facing_transform(self.get_ground_height(idx)))
        R_sim_inv = R_sim.transpose()
        state_sim = self.state_body(idx, agent="sim", return_stacked=False)
        
        state = []
        state_kin_orig = kin_agent.save_states()
        for pose, vel in zip(poses, vels):
            kin_agent.set_pose(pose, vel)
            state_kin = self.state_body(idx, agent="kin", return_stacked=False)
            # Add pos/vel values
            if include_abs:
                state.append(np.hstack(state_kin))
            # Add difference of pos/vel values
            if include_rel:
                for j in range(len(state_sim)):
                    if np.isscalar(state_sim[j]) or len(state_sim[j])==3 or len(state_sim[j])==6: 
                        state.append(state_sim[j]-state_kin[j])
                    elif len(state_sim[j])==4:
                        state.append(
                            self._pb_client.getDifferenceQuaternion(state_sim[j], state_kin[j]))
                    else:
                        raise NotImplementedError
            ''' Add facing frame differences '''
            R_kin, p_kin = conversions.T2Rp(
                kin_agent.get_facing_transform(self.get_ground_height(idx)))
            state.append(np.dot(R_sim_inv, p_kin - p_sim))
            state.append(np.dot(R_sim_inv, kin_agent.get_facing_direction()))
        kin_agent.restore_states(state_kin_orig)

        return np.hstack(state)

    def state_vr_sensors(
        self,
        idx,
        return_stacked=True):
        
        ref_motion = self._ref_motion[idx] 
        time = self.get_ref_motion_time()
        poses, vels = [], []
        for dt in self._sensor_lookahead:
            t = np.clip(
                time[idx] + dt, 
                0.0, 
                ref_motion.length())
            poses.append(ref_motion.get_pose_by_time(t))
            vels.append(ref_motion.get_velocity_by_time(t))

        sim_agent = self._sim_agent[idx]
        kin_agent = self._kin_agent[idx]

        R_sim, p_sim = conversions.T2Rp(
            sim_agent.get_facing_transform(self.get_ground_height(idx)))
        R_sim_inv = R_sim.transpose()
        state_sim = self.state_body(idx, agent="sim", return_stacked=False)

        state = []
        state_kin_orig = kin_agent.save_states()
        for pose, vel in zip(poses, vels):
            kin_agent.set_pose(pose, vel)
            ps, Qs, vs, ws = kin_agent.get_link_states(indices=self._vr_sensors_link_idx)
            for j in range(len(ps)):
                p, Q = ps[j], Qs[j]
                p_rel = np.dot(R_sim_inv, p - p_sim)
                state.append(p_rel)
                Q_rel = conversions.R2Q(np.dot(R_sim_inv, conversions.Q2R(Q)))
                Q_rel = quaternion.Q_op(Q_rel, op=["normalize", "halfspace"])
                state.append(Q_rel)
        kin_agent.restore_states(state_kin_orig)

        if return_stacked:
            return np.hstack(state)
        else:
            return state
    
    def reward_data(self, idx):
        data = {}

        data['sim_root_pQvw'] = self._sim_agent[idx].get_root_state()
        # data['sim_interaction_graph'] = self.compute_interaction_mesh_lower_triangle(self._sim_agent[idx])

        data['sim_link_pQvw'] = self._sim_agent[idx].get_link_states()
        data['sim_joint_pv'] = self._sim_agent[idx].get_joint_states()
        data['sim_facing_frame'] = self._sim_agent[idx].get_facing_transform(self.get_ground_height(idx))
        data['sim_com'], data['sim_com_vel'] = self._sim_agent[idx].get_com_and_com_vel()
        
        data['kin_root_pQvw'] = self._kin_agent[idx].get_root_state()
        # data['kin_interaction_graph'] = self.compute_interaction_mesh_lower_triangle(self._kin_agent[idx])

        data['kin_link_pQvw'] = self._kin_agent[idx].get_link_states()
        data['kin_joint_pv'] = self._kin_agent[idx].get_joint_states()
        data['kin_facing_frame'] = self._kin_agent[idx].get_facing_transform(self.get_ground_height(idx))
        data['kin_com'], data['kin_com_vel'] = self._kin_agent[idx].get_com_and_com_vel()

        data['sim_sim_interaction_full_graph'] = self.compute_interaction_mesh(self._sim_interaction_points[idx],self._sim_interaction_points[idx])
        data['kin_kin_interaction_full_graph'] = self.compute_interaction_mesh(self._kin_interaction_points[idx],self._kin_interaction_points[idx])
        data['sim_kin_interaction_full_graph'] = self.compute_interaction_mesh(self._sim_interaction_points[idx],self._kin_interaction_points[idx])

        return data
    
    def reward_max(self):
        return 1.0
    
    def reward_min(self):
        return 0.0
    
    def get_task_error(self, idx, data_prev, data_next, actions):
        error = {}

        sim_agent = self._sim_agent[idx]
        char_info = sim_agent._char_info

        data = data_next[idx]

        sim_root_p, sim_root_Q, sim_root_v, sim_root_w = data['sim_root_pQvw']
        sim_link_p, sim_link_Q, sim_link_v, sim_link_w = data['sim_link_pQvw']
        sim_joint_p, sim_joint_v = data['sim_joint_pv']
        sim_facing_frame = data['sim_facing_frame']
        # sim_iteration_graph = data['sim_interaction_graph']

        R_sim_f, p_sim_f = conversions.T2Rp(sim_facing_frame)
        R_sim_f_inv = R_sim_f.transpose()
        sim_com, sim_com_vel = data['sim_com'], data['sim_com_vel']
        
        kin_root_p, kin_root_Q, kin_root_v, kin_root_w = data['kin_root_pQvw']
        kin_link_p, kin_link_Q, kin_link_v, kin_link_w = data['kin_link_pQvw']
        kin_joint_p, kin_joint_v = data['kin_joint_pv']
        kin_facing_frame = data['kin_facing_frame']
        # kin_iteration_graph = data['kin_interaction_graph']

        sim_sim_interaction_full_graph = data['sim_sim_interaction_full_graph']
        sim_kin_interaction_full_graph = data['sim_kin_interaction_full_graph']
        kin_kin_interaction_full_graph = data['kin_kin_interaction_full_graph']

        R_kin_f, p_kin_f = conversions.T2Rp(kin_facing_frame)
        R_kin_f_inv = R_kin_f.transpose()
        kin_com, kin_com_vel = data['kin_com'], data['kin_com_vel']

        indices = range(len(sim_joint_p))

        if self.exist_rew_fn_subterm(idx, 'pose_pos'):
            error['pose_pos'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                elif joint_type == self._pb_client.JOINT_SPHERICAL:
                    dQ = self._pb_client.getDifferenceQuaternion(sim_joint_p[j], kin_joint_p[j])
                    _, diff_pose_pos = self._pb_client.getAxisAngleFromQuaternion(dQ)
                else:
                    diff_pose_pos = sim_joint_p[j] - kin_joint_p[j]
                error['pose_pos'] += char_info.joint_weight[j] * np.dot(diff_pose_pos, diff_pose_pos)
            if len(indices) > 0:
                error['pose_pos'] /= len(indices)

        if self.exist_rew_fn_subterm(idx, 'pose_vel'):
            error['pose_vel'] = 0.0
            for j in indices:
                joint_type = sim_agent.get_joint_type(j)
                if joint_type == self._pb_client.JOINT_FIXED:
                    continue
                else:
                    diff_pose_vel = sim_joint_v[j] - kin_joint_v[j]
                error['pose_vel'] += char_info.joint_weight[j] * np.dot(diff_pose_vel, diff_pose_vel)
            if len(indices) > 0:
                error['pose_vel'] /= len(indices)

        if self.exist_rew_fn_subterm(idx, 'ee'):
            error['ee'] = 0.0
            
            for j in char_info.end_effector_indices:
                sim_ee_local = np.dot(R_sim_f_inv, sim_link_p[j]-p_sim_f)
                kin_ee_local = np.dot(R_kin_f_inv, kin_link_p[j]-p_kin_f)
                diff_pos =  sim_ee_local - kin_ee_local
                error['ee'] += np.dot(diff_pos, diff_pos)

            if len(char_info.end_effector_indices) > 0:
                error['ee'] /= len(char_info.end_effector_indices)

        if self.exist_rew_fn_subterm(idx, 'root'):
            diff_root_p = sim_root_p - kin_root_p
            _, diff_root_Q = self._pb_client.getAxisAngleFromQuaternion(
                self._pb_client.getDifferenceQuaternion(sim_root_Q, kin_root_Q))
            diff_root_v = sim_root_v - kin_root_v
            diff_root_w = sim_root_w - kin_root_w
            error['root'] = 1.0 * np.dot(diff_root_p, diff_root_p) + \
                            0.1 * np.dot(diff_root_Q, diff_root_Q) + \
                            0.01 * np.dot(diff_root_v, diff_root_v) + \
                            0.001 * np.dot(diff_root_w, diff_root_w)

        if self.exist_rew_fn_subterm(idx, 'com'):
            diff_com = np.dot(R_sim_f_inv, sim_com-p_sim_f) - np.dot(R_kin_f_inv, kin_com-p_kin_f)
            diff_com_vel = np.dot(R_sim_f_inv, sim_com_vel) - np.dot(R_kin_f_inv, kin_com_vel)
            error['com'] = 1.0 * np.dot(diff_com, diff_com) + \
                           0.1 * np.dot(diff_com_vel, diff_com_vel)

        if self.exist_rew_fn_subterm(idx, 'im_pos'):
            error['im_pos'] = 0
            diff = np.array(sim_sim_interaction_full_graph)[:,:,:3]-np.array(kin_kin_interaction_full_graph)[:,:,:3]
            per_joint_dist = np.linalg.norm(diff,axis=2)**2
            error['im_pos'] = np.mean(per_joint_dist)
            # print("IM pos error: %f, IM pos rwd with scale 20: %f"%(error['im_pos'],np.exp(-10*error['im_pos'])))
        if self.exist_rew_fn_subterm(idx, 'im_vel'):
            error['im_vel'] = 0
            diff = np.array(sim_sim_interaction_full_graph)[:,:,3:]-np.array(kin_kin_interaction_full_graph)[:,:,3:]
            per_joint_vel_dist = np.linalg.norm(diff,axis=2)**2
            error['im_vel'] = np.mean(per_joint_vel_dist)

        if self.exist_rew_fn_subterm(idx,'im_sim_kin_pos_diff'):
            error['im_sim_kin_pos_diff'] = 0.0
            dist = np.linalg.norm(sim_sim_interaction_full_graph[:,:,:3]-sim_kin_interaction_full_graph[:,:,:3],axis=2)**2
            error['im_sim_kin_pos_diff'] = np.mean(dist)
            # print("Sim-kin Interaction pos err: %f, Reward with kernel width 20: %f"%(error['im_sim_kin_pos_diff'],np.exp(-40*error['im_sim_kin_pos_diff'])))
        if self.exist_rew_fn_subterm(idx,'im_sim_kin_vel_diff'):
            error['im_sim_kin_vel_diff'] = 0.0
            dist = np.linalg.norm(sim_sim_interaction_full_graph[:,:,3:]-sim_kin_interaction_full_graph[:,:,3:],axis=2)**2
            error['im_sim_kin_vel_diff'] = np.mean(dist)
            # print("Sim-kin Interaction vel err: %f, Reward with kernel width 10: %f"%(error['im_sim_kin_vel_diff'],np.exp(-5*error['im_sim_kin_vel_diff'])))
        return error

    def inspect_end_of_episode(self):
        eoe_reason = super().inspect_end_of_episode()
        
        cur_time = self.get_current_time()
        for i in range(self._num_agent):
            name = self._sim_agent[i].get_name()
            if "ref_motion_end" in self._early_term_choices:
                check = cur_time[i] >= self._ref_motion[i].length()
                if check: eoe_reason.append('[%s] end_of_motion'%self._sim_agent[i].get_name())    
            if "root_mismatch_orientation" in self._early_term_choices or \
               "root_mismatch_position" in self._early_term_choices:
                p1, Q1, v1, w1 = self._sim_agent[i].get_root_state()
                p2, Q2, v2, w2 = self._kin_agent[i].get_root_state()
                ''' TODO: remove pybullet Q utils '''
                if "root_mismatch_orientation" in self._early_term_choices:
                    dQ = self._pb_client.getDifferenceQuaternion(Q1, Q2)
                    _, diff = self._pb_client.getAxisAngleFromQuaternion(dQ)
                    # Defalult threshold is 60 degrees
                    thres = self._config['early_term'].get('root_mismatch_orientation_thres', 1.0472)
                    check = diff > thres
                    if check: eoe_reason.append('[%s]:root_mismatch_orientation'%name)
                if "root_mismatch_position" in self._early_term_choices:
                    diff = np.linalg.norm(p2-p1)
                    thres = self._config['early_term'].get('root_mismatch_position_thres', 0.5)
                    check = diff > thres
                    if check: eoe_reason.append('[%s]:root_mismatch_position'%name)
        return eoe_reason        

    def get_ref_motion_time(self):
        cur_time = self.get_current_time()
        if self._repeat_ref_motion:
            for i in range(self._num_agent):
                motion_length = self._ref_motion[i].length()
                while cur_time[i] > motion_length:
                    cur_time[i] -= motion_length
        return cur_time

    def get_current_time(self):
        return self._start_time + self.get_elapsed_time()

    def sample_ref_motion(self, indices=None):
        ref_indices = []
        ref_motions = []
        for i in range(self._num_agent):
            if indices is None:
                idx = np.random.randint(len(self._ref_motion_all[i]))
            else:
                idx = indices[i]
            ref_indices.append(idx)
            ref_motions.append(self._ref_motion_all[i][idx])
        if self._verbose:
            print('Ref. motions selected: ', ref_indices)
        return ref_motions, ref_indices

    def get_phase(self, motion, elapsed_time, mode='linear', **kwargs):
        if mode == 'linear':
            return elapsed_time / motion.length()
        elif mode == 'trigon':
            period = kwargs.get('period', 1.0)
            theta = 2*np.pi * elapsed_time / period
            return np.array([np.cos(theta), np.sin(theta)])
        else:
            raise NotImplementedError

    def check_end_of_motion(self, idx):
        cur_time = self.get_current_time()
        return cur_time[idx] >= self._ref_motion[idx].length()

    def render(self, rm):
        super().render(rm)

        if self._vr_sensors_link_idx:
            sensors = self.state_vr_sensors(0, False)
            T_face = self._sim_agent[0].get_facing_transform(self.get_ground_height(0))
            for i in range(len(sensors)//2):
                p, Q = sensors[2*i], sensors[2*i+1]
                T = np.dot(T_face, conversions.Qp2T(Q, p))
                rm.gl_render.render_transform(T, scale=0.25, use_arrow=True)
        if rm.flag['custom1']:
            for i in range(self._num_agent):
                if rm.flag['sim_model']:
                    sim_interaction_point = self._sim_interaction_points[i]
                    for state in sim_interaction_point:
                        p = state[:3]
                        v = state[3:]
                        rm.gl.glPushMatrix()
                        rm.gl.glTranslatef(p[0],p[1],p[2])
                        rm.gl.glScalef(0.14,0.14,0.14)
                        rm.gl_render.render_sphere(
                            constants.EYE_T, 0.4, color=[1, 0, 0, 1], slice1=16, slice2=16)
                        rm.gl.glPopMatrix()
                        rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0.5, 0.5, 0.5, 1])

                if rm.flag['kin_model']:
                    kin_interaction_point = self._kin_interaction_points[i]
                    for state in kin_interaction_point:
                        p = state[:3]
                        v = state[3:]
                        rm.gl.glPushMatrix()
                        rm.gl.glTranslatef(p[0],p[1],p[2])
                        rm.gl.glScalef(0.14,0.14,0.14)
                        rm.gl_render.render_sphere(
                            constants.EYE_T, 0.4, color=[1, 0, 0, 1], slice1=16, slice2=16)
                        rm.gl.glPopMatrix()
                        rm.gl_render.render_arrow(p, p+v, D=0.01, color=[0.5, 0.5, 0.5, 1])
        if rm.flag['custom2']:
            for i in range(self._num_agent):
                if rm.flag['sim_model']:
                    sim_interaction_points = self._sim_interaction_points[i]
                    sim_interaction_graph = self.compute_interaction_mesh(sim_interaction_points,sim_interaction_points)
                    
                    for k in range(len(sim_interaction_points)):
                        interactions = sim_interaction_graph[k]
                        p = sim_interaction_points[k][:3]
                        for interaction in interactions:
                            pos_diff = interaction[:3]
                            target_pos = p + pos_diff
                            rm.gl_render.render_line(p, target_pos, color=[1, 0, 0, 0.6])
                    

                if rm.flag['kin_model']:
                    kin_interaction_points = self._kin_interaction_points[i]
                    kin_interaction_graph = self.compute_interaction_mesh(kin_interaction_points,kin_interaction_points)
                    
                    for k in range(len(kin_interaction_points)):
                        interactions = kin_interaction_graph[k]
                        p = kin_interaction_points[k][:3]
                        for interaction in interactions:
                            pos_diff = interaction[:3]
                            target_pos = p + pos_diff
                            rm.gl_render.render_line(p, target_pos, color=[1, 0, 0, 0.6])
                    
                
        if rm.flag['custom3'] and self.current_interaction:
            
            interaction_mesh = self.current_interaction
            for i in range(self._num_agent):
                edge_index = interaction_mesh[i]
                if rm.flag['sim_model']:
                    sim_interaction_points = self._sim_interaction_points[i]
                    pa = sim_interaction_points[edge_index[0]]
                    pb =  sim_interaction_points[edge_index[1]]
                    for k in range(len(pa)):
                        rm.gl_render.render_line(pa[k], pb[k], color=[0, 0, 1, 0.6])

                if rm.flag['kin_model']:
                        kin_interaction_points = self._kin_interaction_points[i]
                        pa = kin_interaction_points[edge_index[0]]
                        pb =  kin_interaction_points[edge_index[1]]
                        for k in range(len(pa)):
                            rm.gl_render.render_line(pa[k], pb[k], color=[0, 0, 1, 0.6])

if __name__ == '__main__':

    import env_renderer as er
    import render_module as rm
    import argparse
    from fairmotion.viz.utils import TimeChecker

    rm.initialize()
    
    def arg_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True, type=str)
        return parser

    class EnvRenderer(er.EnvRenderer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.time_checker_auto_play = TimeChecker()
            self.reset()
        def reset(self):
            self.env.reset({})
        def one_step(self):
            # a = np.zeros(100)
            self.env.step()
        def extra_render_callback(self):
            if self.rm.flag['follow_cam']:
                p, _, _, _ = env._sim_agent[0].get_root_state()
                self.rm.viewer.update_target_pos(p, ignore_z=True)
            self.env.render(self.rm)
        def extra_idle_callback(self):
            time_elapsed = self.time_checker_auto_play.get_time(restart=False)
            if self.rm.flag['auto_play'] and time_elapsed >= self.env._dt_act:
                self.time_checker_auto_play.begin()
                self.one_step()
        def extra_keyboard_callback(self, key):
            if key == b'r':
                self.reset()
            elif key == b'O':
                size = np.random.uniform(0.1, 0.3, 3)
                p, Q, v, w = self.env._agent[0].get_root_state()
                self.env._obs_manager.throw(p, size=size)
    
    print('=====Humanoid Imitation Environment=====')
    
    args = arg_parser().parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    env = Env(config['config']['env_config'])

    cam = rm.camera.Camera(pos=np.array([12.0, 0.0, 12.0]),
                           origin=np.array([0.0, 0.0, 0.0]), 
                           vup=np.array([0.0, 0.0, 1.0]), 
                           fov=30.0)

    renderer = EnvRenderer(env=env, cam=cam)
    renderer.run()
