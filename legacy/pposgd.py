from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
import pickle

from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

import math
from basecode.utils import basics
from basecode.math import mcmc

import policy
import shutil 

GRAD_CLIP_NORM = None
ADAPTIVE_INIT_SAMPLING = False
ADAPTIVE_INIT_RECON_PDF = False
ADAPTIVE_INIT_RESET_CNT = 1

# PBV: progressive body variation
TRAIN_OPTION = {}
TRAIN_OPTION['method'] = []
TRAIN_OPTION['val_mar'] = False
TRAIN_OPTION['val_mar_update_iter'] = 1
##XXX
# TRAIN_OPTION['method'] = ['EgoNet', 'PBV']
# TRAIN_OPTION['method'] = ['MCMC']
# TRAIN_OPTION['method'] = ['SIMPLE']

if 'SIMPLE' in TRAIN_OPTION['method']:
    TRAIN_OPTION['val_mar'] = True
    TRAIN_OPTION['val_mar_update_iter'] = 1
    TRAIN_OPTION['update_pool_iter'] = 20
    TRAIN_OPTION['pool_size'] = 1000
    TRAIN_OPTION['v_update_size'] = 500
    TRAIN_OPTION['k_min'] = 0
    TRAIN_OPTION['k_max'] = 15
    TRAIN_OPTION['k_max_iter'] = 1000

if 'MCMC' in TRAIN_OPTION['method']:
    TRAIN_OPTION['val_mar'] = True
    TRAIN_OPTION['mcmc_sampling_start_iter'] = 1
    TRAIN_OPTION['val_mar_update_iter'] = 1

if 'EgoNet' in TRAIN_OPTION['method']:
    TRAIN_OPTION['egonet_id_loss'] = False
    TRAIN_OPTION['egonet_id_loss_maxiter'] = 300

if 'PBV' in TRAIN_OPTION['method']:
    TRAIN_OPTION['pbv_max_iter'] = 600

def get_cur_k(iters):
    min_k = TRAIN_OPTION['k_min']
    max_k = TRAIN_OPTION['k_max']
    dk = max_k - min_k
    return min(max_k, dk * float(iters)/TRAIN_OPTION['k_max_iter'] + min_k)

class InitStateSamplerSimple(object):
    def __init__(self, v_fn, train_param):
        self.v_fn = v_fn
        self.initialized = False
        self.train_param = train_param
        self.train_param_max = np.array([e.val_max for e in self.train_param.entry])
        self.train_param_min = np.array([e.val_min for e in self.train_param.entry])
        self.k = 0.0
        self.reset()
    def sample(self, n, additive=False, num_start_pos=10):
        if not additive: del self.pool[:]
        num_added = 0
        for i in range(num_start_pos+1):
            x_cur = self.init_dist()
            for j in range(int(n/num_start_pos)):
                self.pool.append(x_cur.copy())
                x_new = self.proposal_dist(x_cur)
                alpha = min(1.0, self.target_dist(x_new)/self.target_dist(x_cur))
                if np.random.rand() <= alpha:          
                    x_cur = x_new
                num_added += 1
                if num_added >= n: break
        # print(len(self.pool))
    def update_v_mean(self, x, update_factor=0.001):
        denom = 1.0/(self.n_v_mean+1)
        v = self.v_fn(x)
        self.v_mean = self.n_v_mean*denom*self.v_mean + denom*v
        self.v_max = max(self.v_max, v)
        self.v_min = min(self.v_min, v)
        self.n_v_mean += 1
    def get_random_sample_from_pool(self):
        assert len(self.pool) > 0
        idx = self.n_sampled%len(self.pool)
        self.n_sampled += 1
        return self.pool[idx]
    def reset(self):
        self.v_mean = 0.0
        self.v_max = -float('inf')
        self.v_min = float('inf')
        self.n_v_mean = 0
        self.n_sampled = 0
        self.pool = []
    # k: adjustment factor
    def init_dist(self):
        x = [np.random.uniform(e.val_min, e.val_max) for e in self.train_param.entry]
        return np.array(x)
    def proposal_dist(self, x):
        x_new = [np.random.uniform(e.val_min, e.val_max) for e in self.train_param.entry]
        return np.array(x_new)
    # def proposal_dist(self, x, k=0.005):
    #     mu = x
    #     cov = np.diag(np.array([k*(e.val_max-e.val_min) for e in self.train_param.entry]))
    #     passed = False
    #     for i in range(100):
    #         x_new = np.random.multivariate_normal(mu, cov)
    #         for j in range(self.train_param.num_entry()):
    #             e = self.train_param.get_entry(j)
    #             if not (e.val_min <= x_new[j] <= e.val_max): break
    #         passed = True if j==self.train_param.num_entry()-1 else False
    #         if passed: break
    #     # return x_new
    #     return np.clip(x_new, self.train_param_min, self.train_param_max)
    #     # return np.clip(
    #     #     np.random.multivariate_normal(mu, cov), 
    #     #     self.train_param_min, self.train_param_max)
    def target_dist(self, x):
        val = float(max(1.0e-02, self.v_fn(x)))
        return math.exp(-self.k*(val-self.v_mean)/self.v_mean)
    # def target_dist(self, x, k=10):
    #     val = float(max(1.0e-02, self.v_fn(x)))
    #     return 1.0/(val/self.v_mean)**k
    #     # return math.exp(-k*(val)/self.v_mean)

class InitStateSampler(mcmc.Metropolis):
    def __init__(self, v_fn, x0):
        self.v_fn = v_fn
        self.x0 = x0
        self.vmean_old = 1.0
        self.vmean = 1.0
        self.vmax = -float('inf')
        self.vmin = float('inf')
        self.vmax_full = 0.0
        self.vmin_full = 0.0
        self.initialized = False
        self.pool = []
    def target_dist(self, x):
        val = float(max(1.0, self.v_fn(x)))
        k = 5.0 # adjustment factor
        return math.exp(-k*(val-self.vmean_old)/self.vmean_old)
    # def target_dist(self, x, coeff_pow=1.0):
    #     # val = float(max(1.0e-2, self.v_fn(x)))
    #     val = float(max(1.0, self.v_fn(x)))
    #     return math.pow(1.0/val, coeff_pow)
    # def compute_alpha(self, x_new, x_cur):
    #     val_new = float(max(1.0e-4, self.v_fn(x_new)))
    #     val_cur = float(max(1.0e-4, self.v_fn(x_cur)))
    #     return max(0.05, 1.0 - (val_new/val_cur))
    # def compute_alpha(self, x_new, x_cur):
    #     val_new = float(max(1.0, self.v_fn(x_new)))
    #     val_cur = float(max(1.0, self.v_fn(x_cur)))
    #     d = (1.0 - min(1.0, val_cur/val_new))
    #     sigma = 0.01
    #     return min(1.0, math.exp(-d*d/sigma) + 0.05)
    def update_vmean(self):
        self.vmean_old = self.vmean
    def propose_dist(self):
        return np.random.normal(0.0, 0.1, size=len(self.x0))
    def reset(self):
        self.vmax = -float('inf')
        self.vmin = float('inf')
        self.vmax_full = 0.0
        self.vmin_full = 0.0
        self.pool = []
        super(InitStateSampler, self).reset(self.x0, 0)
    def accept(self, x_new):
        v = self.v_fn(x_new)
        r = 0.001
        self.vmean = (1.0-r)*self.vmean + r*v
        self.vmax = max(self.vmax, v)
        self.vmin = min(self.vmin, v)
        return super(InitStateSampler, self).accept(x_new)
    
class InitStateDistribution(object):
    def __init__(self, mode='exp2'):
        self.vmax = -float('inf')
        self.vmin = float('inf')
        self.mode = mode
        self.initialized = True
    def update(self, value):
        self.vmax = max(self.vmax, value)
        self.vmin = min(self.vmin, value)
    def prob_reject(self, value):
        vmax = self.vmax
        vmin = self.vmin
        if vmax<=vmin:
            return 0.0
        assert vmin <= value <= vmax
        d = (value-vmin)/(vmax-vmin)
        if self.mode=='linear':
            prob = d
        elif self.mode=='arc':
            prob = math.sin(0.5*math.pi*d)
        elif self.mode=='exp1':
            prob = math.exp(-(d-1)*(d-1)/0.1)
        elif self.mode=='exp2':
            prob = 1.0-math.exp(-d*d/0.1)
        else:
            raise NotImplementedError
        return prob
    def reject(self, value):
        return not self.accept()
    def accept(self, value):
        v = np.random.uniform() + 1e-2
        # print(v, self.prob_reject(value))
        return v >= self.prob_reject(value)
    def reset(self):
        self.vmax = -float('inf')
        self.vmin = float('inf')

def traj_segment_generator(pi, env, horizon, stochastic, file_record_dir, rho=None, cluster_based_train=False):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()    

    #DEBUG
    reward_keys = env.env.reward_keys
    cur_ep_ret_detail = {}
    ep_rets_detail = {}
    for key in reward_keys:
        cur_ep_ret_detail[key] = 0
        ep_rets_detail[key] = []
    #END

    while True:
        '''
        This is for cluster-based learning
        '''
        if cluster_based_train:
            assert env.cluster_id_assigned >= 0
            if env.force_reset: 
                env.force_reset = False
                new = True
                for key in reward_keys:
                    ep_rets_detail[key].append(cur_ep_ret_detail[key])
                    cur_ep_ret_detail[key] = 0
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()
            if env.train_experts:
                prevac = ac
                ac, vpred = pi.act_expert(stochastic, ob, env.cluster_id_assigned)
            else:
                prevac = ac
                ac, vpred = pi.act(stochastic, ob)
        else:
            prevac = ac
            ac, vpred = pi.act(stochastic, ob)
        
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                   "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets" : ep_rets, "ep_lens" : ep_lens,
                   "ep_rets_detail" : ep_rets_detail}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            #DEBUG
            for key in reward_keys:
                ep_rets_detail[key] = []
            #END
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        # print(vpreds_param[i], vpreds[i])

        # assert basics.check_valid_data(ob)
        # assert basics.check_valid_data(vpred)
        # assert basics.check_valid_data(new)
        # assert basics.check_valid_data(ac)
        # assert basics.check_valid_data(prevac)

        ob, rew, new, info = env.step(ac)
        rews[i] = rew

        # assert basics.check_valid_data(rew)

        if 'sim_div' in info['eoe_reason']:
            print('**********One step rollback**********')
            with open(os.path.join(file_record_dir, \
                "sim_div_%d.txt"%MPI.COMM_WORLD.Get_rank()), "a") as f:
                f.write("-------------------------")
                f.write(obs)
                f.write("-------------------------")
            ac = prevac
            t -= 1
            new = True
        else:
            #DEBUG
            for key in reward_keys:
                e = info['rew_detail'][key][0]
                w = info['rew_detail'][key][1]
                cur_ep_ret_detail[key] += w * e
            #END
            cur_ep_ret += rew
            cur_ep_len += 1
        
        if new:
            #DEBUG
            for key in reward_keys:
                ep_rets_detail[key].append(cur_ep_ret_detail[key])
                cur_ep_ret_detail[key] = 0
            # END
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def flatgrad(loss, var_list, clip_value_min=None, clip_value_max=None):
    grads = tf.gradients(loss, var_list)
    if clip_value_min is not None and clip_value_max is not None:
        grads = [tf.clip_by_value(grad, clip_value_min, clip_value_max) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [U.numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])

def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, vfcoeff, # clipping parameter epsilon, entropy coeff, vfloss coeff
        optim_epochs, optim_stepsize_pol, optim_stepsize_val, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        mode='train',
        network_number=None, 
        network_number_vmar=None, 
        log_learning_curve=None,
        file_record_period=10,
        file_record_dir="data/learning",
        policy_name="ego",
        w_new_expert_usage=0.0,
        gate_expert_alter=False,
        gate_expert_alter_gate_iter=10,
        gate_expert_alter_expert_iter=40,
        ob_filter_update_for_expert=True,
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_fn("pi/%s"%policy_name, ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi/%s"%policy_name, ob_space, ac_space) # Network for old policy

    # Load Variables
    if mode=='load' or mode=='retrain':
        # assert network_number is not None
        if network_number is not None:
            name = file_record_dir + '/network' + str(network_number)
            pi.load_variables(name)
            print('Network Loaded:', name)
        else:
            mode = "test"
        if network_number_vmar is not None:
            name = file_record_dir + '/network_vmar' + str(network_number_vmar)
            val_mar.load_variables(name)
            print('Network (V_mar) Loaded:', name)
        if mode == 'load':
            return pi

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = pi.input
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = (vfcoeff) * tf.reduce_mean(tf.square(pi.vpred - ret))

    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    if gate_expert_alter:
        pol_surr_experts, pol_entpen_experts, vf_loss_experts, meankl_experts, meanent_experts = [], [], [], [], []
        for i in range(pi.num_experts()):
            oldpi_expert = oldpi.experts[i]
            pi_expert = pi.experts[i]
            
            meankl_expert = tf.reduce_mean(oldpi_expert.pd.kl(pi_expert.pd))
            meanent_expert = tf.reduce_mean(pi_expert.pd.entropy())
            pol_entpen_expert = (-entcoeff) * meanent_expert

            ratio_expert = tf.exp(pi_expert.pd.logp(ac) - oldpi_expert.pd.logp(ac))
            surr1_expert = ratio_expert * atarg 
            surr2_expert = tf.clip_by_value(ratio_expert, 1.0 - clip_param, 1.0 + clip_param) * atarg #
            pol_surr_expert = - tf.reduce_mean(tf.minimum(surr1_expert, surr2_expert))
            vf_loss_expert = (vfcoeff) * tf.reduce_mean(tf.square(pi_expert.vpred - ret))

            pol_surr_experts.append(pol_surr_expert)
            pol_entpen_experts.append(pol_entpen_expert)
            vf_loss_experts.append(vf_loss_expert)
            meankl_experts.append(meankl_expert)
            meanent_experts.append(meanent_expert)

        total_loss_experts = []
        losses_experts = []
        for i in range(pi.num_experts()):
            total_loss_experts.append(pol_surr_experts[i] + pol_entpen_experts[i] + vf_loss_experts[i])
            losses_experts.append([pol_surr_experts[i],
                                   pol_entpen_experts[i], 
                                   vf_loss_experts[i], 
                                   meankl_experts[i], 
                                   meanent_experts[i]])
        loss_names_expert = ["pol_surr_expert", "pol_entpen_expert", "vf_loss_expert", "kl_expert", "ent_expert"]

    ''' We encourage the use of new experts '''
    if isinstance(pi, (policy.AdditivePolicy, policy.MultiplicativePolicy, policy.MultiplicativePolicy2)):
        if w_new_expert_usage > 0.0:
            new_expert_usage = w_new_expert_usage * tf.reduce_mean(1.0 - tf.reduce_sum(pi.weights_for_new, 1))
            total_loss = total_loss + new_expert_usage
            losses.append(new_expert_usage)
            loss_names.append("new_expt_use")

    var_list_pol = pi.get_trainable_variables(ac=True, vf=False)
    var_list_val = pi.get_trainable_variables(ac=False, vf=True)

    # print('**********************************')
    # print(var_list_pol)
    # print('**********************************')
    # print(var_list_val)

    if gate_expert_alter:
        assert ob_filter_update_for_expert
        var_list_gate = pi.get_trainable_variables_gate()
        var_list_expert = [pi.get_trainable_variables_experts(i) for i in range(pi.num_experts())]
        var_list_val = pi.get_trainable_variables(ac=False, vf=True, include_experts=False)
    
    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables_all(), pi.get_variables_all())])

    ph_list = [ob, ac, atarg, ret, lrmult]

    lossandgrad_pol = U.function(ph_list, losses + [U.flatgrad(total_loss, var_list_pol, GRAD_CLIP_NORM)])
    lossandgrad_val = U.function(ph_list, losses + [U.flatgrad(total_loss, var_list_val, GRAD_CLIP_NORM)])
    if gate_expert_alter:
        lossandgrad_gate = U.function(ph_list, losses + [U.flatgrad(total_loss, var_list_gate, GRAD_CLIP_NORM)])
        lossandgrad_expert = [U.function(ph_list, losses_experts[i] + [U.flatgrad(total_loss_experts[i], var_list_expert[i], GRAD_CLIP_NORM)])\
            for i in range(pi.num_experts())]

    adam_pol = MpiAdam(var_list_pol, epsilon=adam_epsilon)
    adam_val = MpiAdam(var_list_val, epsilon=adam_epsilon)
    compute_losses = U.function(ph_list, losses)

    if gate_expert_alter:
        adam_gate = MpiAdam(var_list_gate, epsilon=adam_epsilon)
        adam_expert = [MpiAdam(var_list_expert[i], epsilon=adam_epsilon) for i in range(pi.num_experts())]
        compute_losses_expert = [U.function(ph_list, losses_experts[i]) for i in range(pi.num_experts())]

    # This function intializes all variables including non-traiable ones
    U.initialize()

    if isinstance(pi, (policy.AdditivePolicy, policy.MultiplicativePolicy, policy.MultiplicativePolicy2)):
        pi.assign_old_experts()
    
    assign_old_eq_new()

    # This is testing behavior of the policy when it is initialized
    if mode=='test':
        return pi

    adam_pol.sync()
    adam_val.sync()
    if gate_expert_alter:
        adam_gate.sync()
        for i in range(pi.num_experts()):
            adam_expert[i].sync()

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    if mode=='retrain':
        iters_so_far = network_number
        timesteps_so_far = iters_so_far * MPI.COMM_WORLD.Get_size() * timesteps_per_actorbatch
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    rewards_for_save =[]

    rho = None

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(
        pi, env, timesteps_per_actorbatch, stochastic=True, file_record_dir=file_record_dir, rho=rho)

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1, "Only one time constraint permitted"

    # Prepare data for plotting
    reward_keys = env.env.reward_keys
    rewdetailbuffer = {}
    rewdetail_eval = {}
    for key in reward_keys:
        rewdetailbuffer[key] = deque(maxlen=100)
        rewdetail_eval[key] = []
    
    if MPI.COMM_WORLD.Get_rank()==0:
        if log_learning_curve == 'matplot':
            from basecode.plot import plot_evaluation as pe
            rew_eval = []
            plot_eval = pe.PlotEvaluation(id=0,
                                          title=file_record_dir, 
                                          num_data=len(reward_keys)+1, 
                                          label=['total']+reward_keys)
            plot_eval_param = None
            plot_eval_param_orig = None
            plot_eval_sample = None
            
            if TRAIN_OPTION['val_mar']:
                train_param = env.env.train_param
                if train_param.num_entry() >= 2:
                    entry0 = train_param.get_entry(0)
                    entry1 = train_param.get_entry(1)
                    slice_eval_param = 20
                    x_name = entry0.name
                    x_range = np.linspace(entry0.val_min, entry0.val_max, slice_eval_param)      
                    y_name = entry1.name
                    y_range = np.linspace(entry1.val_min, entry1.val_max, slice_eval_param) 
                    X_eval_param, Y_eval_param = np.meshgrid(x_range, y_range)
                    Z_eval_param = np.zeros(X_eval_param.shape)
                    Z_eval_param_orig = np.zeros(X_eval_param.shape)
                    plot_eval_param = pe.Surface3D(id=1, title=file_record_dir, xlabel=x_name, ylabel=y_name)
                    plot_eval_param.set_domain(X_eval_param, Y_eval_param)
                    plot_eval_param.set_codomain_range(0, 600)
                    if "SIMPLE" in TRAIN_OPTION["method"]:
                        #plot_eval_param_orig = pe.Surface3D(id=2, title=file_record_dir+'/orig', xlabel=x_name, ylabel=y_name)
                        #plot_eval_param_orig.set_domain(X_eval_param, Y_eval_param)
                        # plot_eval_sample = pe.Scatter2D(id=3, title=file_record_dir, xlabel=x_name, ylabel=y_name)
                        plot_eval_sample = pe.Histogram3D(id=3, title=file_record_dir, xlabel=x_name, ylabel=y_name)
                        plot_eval_sample.set_range(
                            [entry0.val_min, entry0.val_max], [entry1.val_min, entry1.val_max])
        elif log_learning_curve == "file":
            log_file = open('%s/log.txt'%(file_record_dir), 'w+')
            text = "iter\tsteps\ttotal"
            for key in reward_keys:
                text += "\t%s"%key
            text += "\n"
            log_file.write(text)
            log_file.close()
        elif log_learning_curve == "tensorboard":
            summ_writer = tf.summary.FileWriter(file_record_dir, U.get_session().graph)
            tf_summaries_inputs = []
            with tf.name_scope(file_record_dir+'/reward'):
                tf_rew_ph = tf.placeholder(tf.float32, shape=None, name="total_summary_ph")
                tf_rew_summary = tf.summary.scalar("total", tf_rew_ph)
                tf_summaries_inputs.append(tf_rew_ph)
                for key in reward_keys:
                    tf_rewdetail_ph = tf.placeholder(tf.float32, shape=None, name=key+"_summary_ph")
                    tf_rewdetail_summary = tf.summary.scalar(key, tf_rewdetail_ph)
                    tf_summaries_inputs.append(tf_rewdetail_ph)
            rew_summaries = tf.summary.merge_all()
            run_summaries = U.function(tf_summaries_inputs, [rew_summaries])
    if gate_expert_alter:
        gate_train = True
        while True:
            gate_train = not gate_train
            gate_expert_alter_cnt = gate_expert_alter_gate_iter if gate_train else gate_expert_alter_expert_iter
            if gate_expert_alter_cnt > 0: break
        env.train_experts = not gate_train
        env.force_reset = True
    
    time_check_per_iter = basics.TimeChecker()
    # num_cluster = env.num_cluster
    while True:
        if callback: callback(locals(), globals())

        finished = False
        
        if max_timesteps and timesteps_so_far >= max_timesteps:
            finished = True
        elif max_episodes and episodes_so_far >= max_episodes:
            finished = True
        elif max_iters and iters_so_far >= max_iters:
            finished = True
        
        if max_seconds and time.time() - tstart >= max_seconds:
            finished = True

        if finished:
            if MPI.COMM_WORLD.Get_rank()==0:
                if log_learning_curve == "file":
                    shutil.copy('%s/log.txt'%(file_record_dir), '%s/log_finished.txt'%(file_record_dir))
            break

        logger.log('TimeConstraint - max_seconds(%d/%d) max_timesteps(%d/%d)'\
            %(time.time()-tstart, max_seconds, timesteps_so_far, max_timesteps))

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)
        if gate_expert_alter:
            total_cnt = gate_expert_alter_gate_iter if gate_train else gate_expert_alter_expert_iter
            logger.log(">> AlterCount %i/%i, GateTrain %r"%(gate_expert_alter_cnt, total_cnt, gate_train))

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

         # update running mean/std for policy
        if isinstance(pi, (policy.AdditivePolicy, 
                           policy.MultiplicativePolicy, 
                           policy.MultiplicativePolicy2)):
            if gate_expert_alter:
                if gate_train:
                    update_gate, update_expert = True, False
                else:
                    update_gate, update_expert = False, True
                if update_gate:
                    if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)
                if update_expert:
                    for expert in pi.experts:
                        if hasattr(expert, "ob_rms"): expert.ob_rms.update(ob)
                logger.log('Update Weights/Obfilter: Gate(%d) Expert(%d)'%(update_gate, update_expert))
            else:
                update_gate, update_old_expert, update_new_expert = True, True, True
                update_gate = update_gate and pi.trainable_gate
                update_old_expert = ob_filter_update_for_expert and update_old_expert and pi.trainable_old_expert
                update_new_expert = update_new_expert and pi.trainable_new_expert
                if update_gate:
                    if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)
                if update_old_expert:
                    for expert in pi.old_experts:
                        if hasattr(expert, "ob_rms"): expert.ob_rms.update(ob)
                if update_new_expert:
                    for expert in pi.new_experts:
                        if hasattr(expert, "ob_rms"): expert.ob_rms.update(ob)
                logger.log('Update Weights : Gate(%d) Expert(%d) Beginner(%d)'%(update_gate, pi.trainable_old_expert, pi.trainable_new_expert))
                logger.log('Update Obfilter: Gate(%d) Expert(%d) Beginner(%d)'%(update_gate, update_old_expert, update_new_expert))
        else:
            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)

        assign_old_eq_new() # set old parameter values to new parameter values

        logger.log("Optimizing...")
        
        # # Here we do a bunch of optimization epochs over the data
        # print(optim_stepsize_pol, optim_stepsize_val)
        logger.log(fmt_row(13, loss_names))
        for epoch in range(optim_epochs):
            losses = [] # list of tuples, detail of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                ph_args = [batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult]
                if gate_expert_alter:
                    # step_size_scale = 1.0/pi.num_experts()
                    step_size_scale = 1.0
                    if gate_train:
                        # if epoch==0: adam_gate.check_synced()
                        *newlosses_gate, g_gate = lossandgrad_gate(*ph_args)
                        adam_gate.update(g_gate, optim_stepsize_pol*cur_lrmult*step_size_scale)
                        *newlosses_val, g_val = lossandgrad_val(*ph_args)
                        adam_val.update(g_val, optim_stepsize_val*cur_lrmult*step_size_scale)
                        losses.append(newlosses_val)
                    else:
                        *newlosses_expert, g_expert = lossandgrad_expert[env.cluster_id_assigned](*ph_args)
                        g_expert_zeros = np.zeros_like(g_expert)
                        for i in range(pi.num_experts()):
                            # if epoch==0: adam_expert[i].check_synced()
                            if i==env.cluster_id_assigned:
                                adam_expert[i].update(g_expert, optim_stepsize_pol*cur_lrmult*step_size_scale)
                            else:
                                adam_expert[i].update(g_expert_zeros, optim_stepsize_pol*cur_lrmult*step_size_scale)
                        losses.append(newlosses_expert)
                else:
                    *newlosses_pol, g_pol = lossandgrad_pol(*ph_args)
                    adam_pol.update(g_pol, optim_stepsize_pol*cur_lrmult)
                    *newlosses_val, g_val = lossandgrad_val(*ph_args)
                    adam_val.update(g_val, optim_stepsize_val*cur_lrmult)
                    losses.append(newlosses_val)
            if epoch%1==0:
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

        if gate_expert_alter:
            gate_expert_alter_cnt -= 1
            if gate_expert_alter_cnt <= 0:
                while True:
                    gate_train = not gate_train
                    gate_expert_alter_cnt = gate_expert_alter_gate_iter if gate_train else gate_expert_alter_expert_iter
                    if gate_expert_alter_cnt > 0: break
                env.train_experts = not gate_train
                env.force_reset = True

        logger.log("Evaluating losses...")
        losses = []
        if gate_expert_alter and not gate_train:
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses_expert[env.cluster_id_assigned](*ph_args)
                losses.append(newlosses)
        else:
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(*ph_args)
                losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        #DEBUG
        for key in reward_keys:
            rewdetailbuffer[key].extend(seg["ep_rets_detail"][key])
        #END
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        for key in reward_keys:
            logger.record_tabular("EpRewMean(%s)"%(key), np.mean(rewdetailbuffer[key]))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("TimePerIteration", time_check_per_iter.get_time())
        
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()
            if iters_so_far % file_record_period == 0:
                name = file_record_dir + '/network' + str(iters_so_far)
                print('SAVING NETWORK...' + name)
                U.save_variables(name, pi.get_variables_all())
                if TRAIN_OPTION['val_mar']:
                    name = file_record_dir + '/network_vmar' + str(iters_so_far)
                    U.save_variables(name, val_mar.get_variables_all())
            # Log Learning Curve
            if log_learning_curve == 'matplot':
                rew_eval.append(np.mean(rewbuffer))
                for key in reward_keys:
                    rewdetail_eval[key].append(np.mean(rewdetailbuffer[key]))
                plot_eval.set_data(rew_eval, 0)
                for i in range(len(reward_keys)):
                    plot_eval.set_data(rewdetail_eval[reward_keys[i]], i+1)
                plot_eval.draw()
                if iters_so_far % file_record_period == 0:
                    plot_eval.save('%s/eval_reward_%d.png'%(file_record_dir, iters_so_far))
                # Shape - Val Graph
                if plot_eval_param is not None:
                    for i in range(slice_eval_param):
                        for j in range(slice_eval_param):
                            x = X_eval_param[i][j]
                            y = Y_eval_param[i][j]
                            Z_eval_param[i][j] = val_mar.val(np.array([x,y]))
                    plot_eval_param.set_value(Z_eval_param)
                    plot_eval_param.draw()
                    if iters_so_far % file_record_period == 0:
                        plot_eval_param.save('%s/eval_param_%d.png'%(file_record_dir, iters_so_far)) 
                        with open('%s/eval_param_%d.data'%(file_record_dir, iters_so_far), 'wb') as f:
                            data = [plot_eval_param.X, plot_eval_param.Y, plot_eval_param.Z]
                            pickle.dump(data, f, protocol=2)
                            f.close()       
                if plot_eval_param_orig is not None:
                    for i in range(slice_eval_param):
                        for j in range(slice_eval_param):
                            x = X_eval_param[i][j]
                            y = Y_eval_param[i][j]
                            s = env.env.env_copy.reset(state_param=np.array([x,y]))
                            _, vpred = pi.act(False, s)
                            Z_eval_param_orig[i][j] = vpred
                    plot_eval_param_orig.set_value(Z_eval_param_orig)
                    plot_eval_param_orig.draw()
                # Sampled Shape Scatter
                if plot_eval_sample is not None:
                    sampled_x = [state_param[0] for state_param in rho.pool]
                    sampled_y = [state_param[1] for state_param in rho.pool]
                    plot_eval_sample.set_data(sampled_x, sampled_y)
                    plot_eval_sample.draw()
                    if iters_so_far % file_record_period == 0:
                        plot_eval_sample.save('%s/eval_sample_%d.png'%(file_record_dir, iters_so_far))
            elif log_learning_curve == "file":
                log_file = open('%s/log.txt'%(file_record_dir), 'a+')
                text = '%d\t%d\t%.4f'%(iters_so_far, timesteps_so_far, np.mean(rewbuffer))
                for key in reward_keys:
                    text += '\t%.4f'%(np.mean(rewdetailbuffer[key]))
                text += '\n'
                log_file.write(text)
                log_file.close()
            elif log_learning_curve == "tensorboard":
                inputs = [np.mean(rewbuffer)]
                for key in reward_keys:
                    inputs.append(np.mean(rewdetailbuffer[key]))
                summ = run_summaries(*inputs)
                summ_writer.add_summary(summ[0], iters_so_far)
        iters_so_far += 1

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
