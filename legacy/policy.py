from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
from baselines.common import zipsame

# NONLINEAR_UNIT = tf.nn.tanh
NONLINEAR_UNIT = tf.nn.relu

def MLP(name, 
        input, 
        output_dim, 
        hid_size, 
        num_hid_layers, 
        input_nonlinearity=False, 
        output_nonlinearity=True,
        trainable=True,
        nonlinear_fn=tf.nn.relu):
    with tf.variable_scope(name):
        output = nonlinear_fn(input) if input_nonlinearity else input
        for i in range(num_hid_layers):
            output = nonlinear_fn(tf.layers.dense(output, hid_size, name="fc%i"%(i+1), trainable=trainable, kernel_initializer=U.normc_initializer(1.0)))
        output = tf.layers.dense(output, output_dim, name="final", trainable=trainable, kernel_initializer=U.normc_initializer(1.0))
        output = nonlinear_fn(output) if output_nonlinearity else output
        scope = tf.get_variable_scope().name
    return output, scope

# repeated feed of some variables
def MLP_RF(input1, input2, output_dim, hid_size, num_hid_layers, input_nonlinearity=False, output_nonlinearity=True, trainable=True):
    input1_processed = NONLINEAR_UNIT(input1) if input_nonlinearity else input1
    input2_processed = NONLINEAR_UNIT(input2) if input_nonlinearity else input2
    output = tf.concat([input1_processed, input2_processed], axis=1)
    for i in range(num_hid_layers):
        output = NONLINEAR_UNIT(tf.layers.dense(output, hid_size, name="fc%i"%(i+1), trainable=trainable, kernel_initializer=U.normc_initializer(1.0)))
        output = tf.concat([output, input2_processed], axis=1)
    output = tf.layers.dense(output, output_dim, name="final", trainable=trainable, kernel_initializer=U.normc_initializer(1.0))
    output = NONLINEAR_UNIT(output) if output_nonlinearity else output
    scope = tf.get_variable_scope().name
    return output

# separate network for two vars
def MLP_D(name, input1, input2, output_dim, hid_size1, num_hid_layers1, hid_size2, num_hid_layers2, hid_size3, num_hid_layers3, input_nonlinearity=False, output_nonlinearity=True, trainable=True):
    with tf.variable_scope(name):
        output1 = NONLINEAR_UNIT(input1) if input_nonlinearity else input1
        for i in range(num_hid_layers1):
            output1 = NONLINEAR_UNIT(tf.layers.dense(output1, hid_size1, name="fc1%i"%(i+1), trainable=trainable, kernel_initializer=U.normc_initializer(1.0)))
        output2 = NONLINEAR_UNIT(input2) if input_nonlinearity else input2
        for i in range(num_hid_layers2):
            output2 = NONLINEAR_UNIT(tf.layers.dense(output2, hid_size2, name="fc2%i"%(i+1), trainable=trainable, kernel_initializer=U.normc_initializer(1.0)))
        output = tf.concat([output1, output2], axis=1)
        for i in range(num_hid_layers3):
            output = NONLINEAR_UNIT(tf.layers.dense(output, hid_size3, name="fc3%i"%(i+1), trainable=trainable, kernel_initializer=U.normc_initializer(1.0)))
        output = tf.layers.dense(output, output_dim, name="final", trainable=trainable, kernel_initializer=U.normc_initializer(1.0))
        output = NONLINEAR_UNIT(output) if output_nonlinearity else output
        scope = tf.get_variable_scope().name
    return output, scope

def PD(input, ac_space, pdtype, gaussian_fixed_var, trainable):
    if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
        mean = tf.layers.dense(input, pdtype.param_shape()[0]//2, name='final_pd', trainable=trainable, kernel_initializer=U.normc_initializer(0.01))
        logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], trainable=trainable, initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
    else:
        pdparam = tf.layers.dense(input, pdtype.param_shape()[0], name='final_pd', trainable=trainable, kernel_initializer=U.normc_initializer(0.01))
    return pdparam

# The mean of a uniform dist. with interval [a b] is (a+b)/2
# The std of a uniform dist. with interval [a b] is (b-a)/sqrt(12)
# Maximum value we can get is (b-(a+b)/2)/((b-a)/sqrt(12))=sqrt(12)/2=1.732
# The abs(min_std, max_std) should be larger than 1.732 to handle the uniform distribution
def NORM_FILTER(name, input, ob_space, min_std=-3.0, max_std=3.0):
    with tf.variable_scope(name):
        ob_rms = RunningMeanStd(shape=ob_space.shape)
        ob = tf.clip_by_value((input - ob_rms.mean) / ob_rms.std, min_std, max_std)
    return ob, ob_rms

#Shape all feed policy
class SRFPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, ob_dim_dynaenv, ob_dim_body, 
              hid_size, num_hid_layers, gaussian_fixed_var=True, ob_filter=True):
        assert isinstance(ob_space, gym.spaces.Box)
        assert ob_dim_dynaenv+ob_dim_body == ob_space.shape[0]

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        stochastic = tf.placeholder(name='stochastic', dtype=tf.bool, shape=())
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        
        # ob shape: [None, 67]
        if ob_filter:
            obz, self.ob_rms = NORM_FILTER("obfilter", ob, ob_space)
        else:
            obz = ob

        ob_bodyenv, ob_shape = tf.split(obz, [ob_dim_dynaenv, ob_dim_body], axis=1)

        with tf.variable_scope("vf"):
            vpred = MLP_RF(ob_bodyenv, ob_shape, 1, 
                           hid_size, num_hid_layers, 
                           False, False, True)
            self.scope_vf = tf.get_variable_scope().name
        with tf.variable_scope("ac"):
            pdparam = MLP_RF(ob_bodyenv, ob_shape, ac_space.shape[0], 
                             hid_size, num_hid_layers,
                             False, True, True)
            pdparam = PD(pdparam, ac_space, self.pdtype, gaussian_fixed_var, True)
            self.scope_ac = tf.get_variable_scope().name

        self.pd = pdtype.pdfromflat(pdparam)
        self.vpred = vpred[:,0]

        self.state_in = []
        self.state_out = []

        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

        self.input = ob
        self.output_ac = self.pd.mode()
        self.output_vf = vpred
    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables_all(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_variables(self, ac=True, vf=True):
        variables = []
        if ac: variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_ac)
        if vf: variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_vf)
        return variables
    def get_trainable_variables(self, ac=True, vf=True):
        variables = []
        if ac: variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_ac)
        if vf: variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_vf)
        return variables
    def get_initial_state(self):
        return []

#Shape divided policy
class SDPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, ob_dim1, ob_dim2, 
              hid_size1, num_hid_layers1, 
              hid_size2, num_hid_layers2, 
              hid_size3, num_hid_layers3,
              gaussian_fixed_var=True, ob_filter=True):
        assert isinstance(ob_space, gym.spaces.Box)
        assert ob_dim1+ob_dim2 == ob_space.shape[0]

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        stochastic = tf.placeholder(name='stochastic', dtype=tf.bool, shape=())
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        
        # ob shape: [None, 67]
        if ob_filter:
            obz, self.ob_rms = NORM_FILTER("obfilter", ob, ob_space)
        else:
            obz = ob

        ob_bodyenv, ob_shape = tf.split(obz, [ob_dim1, ob_dim2], axis=1)

        vpred, _ = MLP_D("vf", 
                         ob_bodyenv, ob_shape, 1, 
                         hid_size1, num_hid_layers1, 
                         hid_size2, num_hid_layers2, 
                         hid_size3, num_hid_layers3,
                         False, False, True)
        pdparam, _ = MLP_D("ac", 
                           ob_bodyenv, ob_shape, ac_space.shape[0], 
                           hid_size1, num_hid_layers1, 
                           hid_size2, num_hid_layers2, 
                           hid_size3, num_hid_layers3,
                           False, True, True)
        pdparam = PD("ac", pdparam, ac_space, self.pdtype, gaussian_fixed_var, True)

        self.pd = pdtype.pdfromflat(pdparam)
        self.vpred = vpred[:,0]

        self.state_in = []
        self.state_out = []

        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

        self.input = ob
        self.output_ac = self.pd.mode()
        self.output_vf = vpred
    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

class ValueMaginal(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
    def _init(self, ob_space, hid_size, num_hid_layers, trainable=True, ob_filter=True):
        assert isinstance(ob_space, gym.spaces.Box)

        sequence_length = None
        ob = U.get_placeholder(name="ob_shape", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        if ob_filter:
            obz, self.ob_rms = NORM_FILTER("obfilter", ob, ob_space)
        else:
            obz = ob

        with tf.variable_scope('vf'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = NONLINEAR_UNIT(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), trainable=trainable, kernel_initializer=U.normc_initializer(1.0)))
            vpred = tf.layers.dense(last_out, 1, name='final', trainable=trainable, kernel_initializer=U.normc_initializer(1.0))
            self.scope_vf = tf.get_variable_scope().name

        self.vpred = vpred[:,0]
        self._val = U.function([ob], [self.vpred])
        self.input = ob
    def val(self, ob):
        vpred1 = self._val(ob[None])
        return vpred1[0]
    def get_variables_all(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def load_variables(self, file):
        variables = self.get_variables_all()
        U.load_variables(file, variables)
        U.ALREADY_INITIALIZED.update(variables)

class EgoPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, independent, stochastic, ob, ob_space, ac_space, hid_size, num_hid_layers, trainable=True, gaussian_fixed_var=True, ob_filter=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        self.independent = independent
        if self.independent:
            assert stochastic is None and ob is None
            ob = U.get_placeholder(name="ob_ego", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
            stochastic = tf.placeholder(name='stochastic', dtype=tf.bool, shape=())
        else:
            assert stochastic is not None and ob is not None

        if ob_filter:
            obz, self.ob_rms = NORM_FILTER("obfilter", ob, ob_space)
        else:
            obz = ob

        with tf.variable_scope('vf'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = NONLINEAR_UNIT(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), trainable=trainable, kernel_initializer=U.normc_initializer(1.0)))
            vpred = tf.layers.dense(last_out, 1, name='final', trainable=trainable, kernel_initializer=U.normc_initializer(1.0))
            self.scope_vf = tf.get_variable_scope().name

        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = NONLINEAR_UNIT(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), trainable=trainable, kernel_initializer=U.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', trainable=trainable, kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], trainable=trainable, initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', trainable=trainable, kernel_initializer=U.normc_initializer(0.01))
            self.scope_ac = tf.get_variable_scope().name

        self.pd = pdtype.pdfromflat(pdparam)
        self.vpred = vpred[:,0]

        self.state_in = []
        self.state_out = []

        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        if self.independent:        
            self._act = U.function([stochastic, ob], [ac, self.vpred])

        self.input = ob
        self.output_ac = self.pd.mode()
        self.output_vf = vpred
    def act(self, stochastic, ob):
        if self.independent:
            ac1, vpred1 = self._act(stochastic, ob[None])
            return ac1[0], vpred1[0]
        else:
            raise NotImplementedError('dependent mode does not support this')
    def get_variables_all(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_variables(self, ac=True, vf=True):
        variables = []
        if ac: variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_ac)
        if vf: variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_vf)
        return variables
    def get_trainable_variables(self, ac=True, vf=True):
        variables = []
        if ac: variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_ac)
        if vf: variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_vf)
        return variables
    def get_initial_state(self):
        return []

class ExoPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
    def _init(self, 
              ob_space_exo, ac_space_exo, hid_size_exo, num_hid_layers_exo, 
              ob_space_ego, ac_space_ego, hid_size_ego, num_hid_layers_ego, 
              gaussian_fixed_var=True, ob_filter=True):

        assert isinstance(ob_space_exo, gym.spaces.Box) and isinstance(ac_space_exo, gym.spaces.Box)
        assert isinstance(ob_space_ego, gym.spaces.Box) and isinstance(ac_space_ego, gym.spaces.Box)
        # assert hid_size_exo > 0 and num_hid_layers_exo > 0 and hid_size_ego > 0 and num_hid_layers_ego > 0

        ob_dim_exo = ob_space_exo.shape[0]
        ob_dim_ego = ob_space_ego.shape[0]
        ob_dim_dynaenv = ob_dim_ego
        ob_dim_body = ob_dim_exo - ob_dim_ego

        assert ob_dim_body > 0

        self.pdtype = pdtype = make_pdtype(ac_space_exo)
        sequence_length = None

        stochastic = tf.placeholder(name='stochastic', dtype=tf.bool, shape=())
        ob_exo = U.get_placeholder(name="ob_exo", dtype=tf.float32, shape=[sequence_length] + list(ob_space_exo.shape))
        # ob_exo shape: [None, 67]
        if ob_filter:
            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=ob_space_exo.shape)
            ob = tf.clip_by_value((ob_exo - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        else:
            ob = ob_exo

        ob_exo_bodyenv, ob_exo_shape = tf.split(ob, [ob_dim_dynaenv, ob_dim_body], axis=1)
        ob_ego, self.scope_ob_exo_to_ob_ego =\
            MLP("ob_exo_to_ob_ego", ob, ob_dim_ego, hid_size_exo, num_hid_layers_exo, False, False, trainable=True)

        self.ego_direct = EgoPolicy(independent=False,
            name="direct_ego", stochastic=stochastic, ob=ob_exo_bodyenv, ob_space=ob_space_ego, ac_space=ac_space_ego, 
            hid_size=hid_size_ego, num_hid_layers=num_hid_layers_ego, trainable=False)

        self.ego = EgoPolicy(independent=False,
            name="ego", stochastic=stochastic, ob=ob_ego, ob_space=ob_space_ego, ac_space=ac_space_ego, 
            hid_size=hid_size_ego, num_hid_layers=num_hid_layers_ego, trainable=False)

        self.assign_ego_direct_from_ego = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(self.ego_direct.get_variables_all(), self.ego.get_variables_all())])

        output_ac_ego = tf.concat([self.ego.output_ac, ob_exo_shape], axis=1)
        output_vf_ego = tf.concat([self.ego.output_vf, ob_exo_shape], axis=1)

        output_ac_exo, self.scope_ac_ego_to_ac_exo =\
            MLP("ac_exo", output_ac_ego, ac_space_exo.shape[0], hid_size_exo, num_hid_layers_exo, False, True)
        output_vf_exo, self.scope_vf_ego_to_vf_exo =\
            MLP("vf_exo", output_vf_ego, 1, hid_size_exo, num_hid_layers_exo, False, False)

        with tf.variable_scope("ac_exo"):
            if gaussian_fixed_var:
                mean = tf.layers.dense(output_ac_exo, pdtype.param_shape()[0]//2, name='final_pd', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(output_ac_exo, pdtype.param_shape()[0], name='final_pd', kernel_initializer=U.normc_initializer(0.01))

        vpred = output_vf_exo

        self.pd = pdtype.pdfromflat(pdparam)
        self.vpred = vpred[:, 0]

        self.state_in = []
        self.state_out = []

        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob_exo], [ac, self.vpred])

        ac_ego_direct = U.switch(stochastic, self.ego_direct.pd.sample(), self.ego_direct.pd.mode())
        self._act_ego_direct = U.function([stochastic, ob_exo], [ac_ego_direct, self.ego_direct.vpred])

        self.input = ob_exo
        self.input_bodyenv = ob_exo_bodyenv
        self.input_shape = ob_exo_shape
        self.output_ac = ac
        self.output_vf = vpred
        self.ob_exo_to_ob_ego = ob_ego
        self.ob_exo_to_ac_ego = self.ego.output_ac
        self.ob_exo_to_vf_ego = self.ego.output_vf
        self.stochastic = stochastic

        self._ob_exo_to_ob_ego = U.function([ob_exo], self.ob_exo_to_ob_ego)
        self._ob_exo_to_ac_ego = U.function([ob_exo], self.ob_exo_to_ac_ego)
        self._ob_exo_to_vf_ego = U.function([ob_exo], self.ob_exo_to_vf_ego)

    def run_ob_exo_to_ob_ego(self, ob):
        ob_ego = self._ob_exo_to_ob_ego(ob[None])
        return ob_ego[0]
    def run_ob_exo_to_ac_ego(self, ob):
        ac_ego = self._ob_exo_to_ac_ego(ob[None])
        return ac_ego[0]
    def run_ob_exo_to_vf_ego(self, ob):
        vf_ego = self._ob_exo_to_vf_ego(ob[None])
        return vf_ego[0]
    def act_ego_direct(self, stochastic, ob):
        ac1, vpred1 = self._act_ego_direct(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables_all(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_variables(self, ob=True, ac=True, vf=True):
        variables = []
        if ob: variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_ob_exo_to_ob_ego)
        if ac: variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_ac_ego_to_ac_exo)
        if vf: variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_vf_ego_to_vf_exo)
        return variables
        # return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self, ob=True, ac=True, vf=True):
        variables = []
        if ob: variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_ob_exo_to_ob_ego)
        if ac: variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_ac_ego_to_ac_exo)
        if vf: variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_vf_ego_to_vf_exo)
        return variables
        # return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []    
    def load_ego(self, load_path):
        variables = self.ego.get_variables_all()
        U.load_variables(load_path, variables)
        self.assign_ego_direct_from_ego()
    def save_ego(self, save_path):
        variables = self.ego.get_variables_all()
        U.save_variables(save_path, variables)
    def load_exo(self, load_path, ob=True, ac=True, vf=True):
        variables = self.get_variables(ob, ac, vf)
        U.load_variables(load_path, variables)
    def save_exo(self, save_path, ob=True, ac=True, vf=True):
        variables = self.get_variables(ob, ac, vf)
        U.save_variables(save_path, variables)

# ac, vf are separate
class ExoPolicy2(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
    def _init(self, 
              ob_space_exo, ac_space_exo, hid_size_exo, num_hid_layers_exo, 
              ob_space_ego, ac_space_ego, hid_size_ego, num_hid_layers_ego, 
              gaussian_fixed_var=True, ob_filter=True):
        assert isinstance(ob_space_exo, gym.spaces.Box) and isinstance(ac_space_exo, gym.spaces.Box)
        assert isinstance(ob_space_ego, gym.spaces.Box) and isinstance(ac_space_ego, gym.spaces.Box)
        # assert hid_size_exo > 0 and num_hid_layers_exo > 0 and hid_size_ego > 0 and num_hid_layers_ego > 0

        ob_dim_exo = ob_space_exo.shape[0]
        ob_dim_ego = ob_space_ego.shape[0]
        ob_dim_dynaenv = ob_dim_ego
        ob_dim_body = ob_dim_exo - ob_dim_ego

        assert ob_dim_body > 0

        self.pdtype = pdtype = make_pdtype(ac_space_exo)
        sequence_length = None

        stochastic = tf.placeholder(name='stochastic', dtype=tf.bool, shape=())
        ob_exo = U.get_placeholder(name="ob_exo", dtype=tf.float32, shape=[sequence_length] + list(ob_space_exo.shape))
        
        # ob_exo shape: [None, 67]
        if ob_filter:
            ob, self.ob_rms = NORM_FILTER("obfilter", ob_exo, ob_space_exo)
        else:
            ob = ob_exo

        ob_exo_dynaenv, ob_exo_body = tf.split(ob, [ob_dim_dynaenv, ob_dim_body], axis=1)

        # Frontend exo_to_ego
        ob_ego, self.scope_ob_exo_to_ob_ego =\
            MLP("ob_exo_to_ob_ego_ac", ob, ob_dim_ego, hid_size_exo, num_hid_layers_exo, False, False, trainable=True)

        # Prepare ego
        self.ego_direct = EgoPolicy(independent=False,
            name="ego", stochastic=stochastic, ob=ob_exo_dynaenv, ob_space=ob_space_ego, ac_space=ac_space_ego, 
            hid_size=hid_size_ego, num_hid_layers=num_hid_layers_ego, trainable=False)

        self.ego = EgoPolicy(independent=False,
            name="ac_ego", stochastic=stochastic, ob=ob_ego, ob_space=ob_space_ego, ac_space=ac_space_ego, 
            hid_size=hid_size_ego, num_hid_layers=num_hid_layers_ego, trainable=False)

        self.assign_ego_from_ego_direct = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(self.ego.get_variables_all(), self.ego_direct.get_variables_all())])

        # Backend ego_to_exo
        output_ac_ego = tf.concat([self.ego.output_ac, ob_exo_body], axis=1)
        output_ac_exo, self.scope_ac_ego_to_ac_exo =\
            MLP("ac_exo", output_ac_ego, ac_space_exo.shape[0], hid_size_exo, num_hid_layers_exo, False, True)
        with tf.variable_scope("ac_exo"):
            pdparam = PD(output_ac_exo, ac_space_exo, self.pdtype, gaussian_fixed_var, True)

        # Normal vf
        output_vf, self.scope_vf = MLP("vf", ob, 1, hid_size_ego, num_hid_layers_ego, False, True)

        # Misc
        vpred = output_vf
        self.pd = pdtype.pdfromflat(pdparam)
        self.vpred = vpred[:, 0]

        self.state_in = []
        self.state_out = []

        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob_exo], [ac, self.vpred])

        ac_ego_direct = U.switch(stochastic, self.ego_direct.pd.sample(), self.ego_direct.pd.mode())
        self._act_ego_direct = U.function([stochastic, ob_exo], [ac_ego_direct, self.ego_direct.vpred])

        self.input = ob_exo
        self.input_dynaenv = ob_exo_dynaenv
        self.input_body = ob_exo_body
        self.output_ac = ac
        self.output_vf = vpred
        self.ob_exo_to_ob_ego = ob_ego
        self.ob_exo_to_ac_ego = self.ego_direct.output_ac
        self.ob_exo_to_vf_ego = self.ego_direct.output_vf
        self.stochastic = stochastic
    def act_ego_direct(self, stochastic, ob):
        ac1, vpred1 = self._act_ego_direct(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables_all(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_variables(self, ac=True, vf=True):
        variables = []
        if ac:
            variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_ob_exo_to_ob_ego)
            variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_ac_ego_to_ac_exo)
        if vf:
            variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_vf)
        return variables
    def get_trainable_variables(self, ac=True, vf=True):
        variables = []
        if ac:
            variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_ob_exo_to_ob_ego)
            variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_ac_ego_to_ac_exo)
        if vf:
            variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_vf)
        return variables
    def get_initial_state(self):
        return []    
    def load_ego(self, load_path):
        variables = self.ego_direct.get_variables_all()
        U.load_variables(load_path, variables)
        # variables = self.ego.get_variables_all()
        # U.load_variables(load_path, variables)
        self.assign_ego_from_ego_direct()
        # self.assign_ego_vf_from_ego_direct()
    def save_ego(self, save_path):
        variables = self.ego_direct.get_variables()
        U.save_variables(save_path, variables)
    # def load_exo(self, load_path, ac=True, vf=True):
    #     variables = self.get_variables(ob, ac, vf)
    #     U.load_variables(load_path, variables)
    # def save_exo(self, save_path, ac=True, vf=True):
    #     variables = self.get_variables(ob, ac, vf)
    #     U.save_variables(save_path, variables)


class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        self._init(name, *args, **kwargs)

    def _init(self, 
              name, 
              independent, 
              stochastic, 
              ob, 
              ob_space, 
              ac_space, 
              hid_size, 
              num_hid_layers, 
              trainable=True, 
              gaussian_fixed_var=True, 
              ob_filter=True, 
              var_reuse=False, 
              prefix_ph='mlp/'):
        
        reuse = tf.AUTO_REUSE if var_reuse else None
        with tf.variable_scope(name, reuse=reuse):
            assert isinstance(ob_space, gym.spaces.Box)

            self.pdtype = pdtype = make_pdtype(ac_space)
            sequence_length = None
            
            self.independent = independent
            if self.independent:
                assert stochastic is None and ob is None
                ob = U.get_placeholder(name=prefix_ph+'ob', dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
                stochastic = tf.placeholder(name=prefix_ph+'stochastic', dtype=tf.bool, shape=())
            else:
                assert stochastic is not None and ob is not None

            if ob_filter:
                obz, self.ob_rms = NORM_FILTER("obfilter", ob, ob_space)
            else:
                obz = ob

            with tf.variable_scope('vf'):
                last_out = obz
                for i in range(num_hid_layers):
                    last_out = NONLINEAR_UNIT(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), trainable=trainable, kernel_initializer=U.normc_initializer(1.0)))
                vpred = tf.layers.dense(last_out, 1, name='final', trainable=trainable, kernel_initializer=U.normc_initializer(1.0))
                self.scope_vf = tf.get_variable_scope().name

            with tf.variable_scope('pol'):
                last_out = obz
                for i in range(num_hid_layers):
                    last_out = NONLINEAR_UNIT(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), trainable=trainable, kernel_initializer=U.normc_initializer(1.0)))
                if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                    mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', trainable=trainable, kernel_initializer=U.normc_initializer(0.01))
                    logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], trainable=trainable, initializer=tf.zeros_initializer())
                    pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                else:
                    pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', trainable=trainable, kernel_initializer=U.normc_initializer(0.01))
                self.scope_ac = tf.get_variable_scope().name

            self.pd = pdtype.pdfromflat(pdparam)
            self.vpred = vpred[:,0]

            ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
            if self.independent:        
                self._act = U.function([stochastic, ob], [ac, self.vpred])

            self.input = ob
            self.output_ac = self.pd.mode()
            self.output_vf = vpred

            self.scope = tf.get_variable_scope().name
    def act(self, stochastic, ob):
        if self.independent:
            ac1, vpred1 = self._act(stochastic, ob[None])
            return ac1[0], vpred1[0]
        else:
            raise NotImplementedError('dependent mode does not support this')
    def get_variables_all(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_variables(self, ac=True, vf=True):
        variables = []
        if ac: variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_ac)
        if vf: variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_vf)
        return variables
    def get_trainable_variables(self, ac=True, vf=True):
        variables = []
        if ac: variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_ac)
        if vf: variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_vf)
        return variables
    def get_initial_state(self):
        return []
    def load_variables(self, file):
        variables = self.get_variables_all()
        U.load_variables(file, variables)
        U.ALREADY_INITIALIZED.update(variables)

class CompositePolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        self._init(name, *args, **kwargs)
    def _init(self, 
              name, 
              old_expert_names, 
              old_expert_weights, 
              independent, 
              stochastic, 
              ob, 
              ob_space, 
              ac_space, 
              hid_size_expert=512,
              num_hid_layers_expert=2,
              hid_size_gate=128,
              num_hid_layers_gate=2,
              new_expert_names=None, 
              ob_filter_gate=True,
              ob_filter_old_expert=True,
              ob_filter_new_expert=True,
              trainable_gate=True,
              trainable_old_expert=False,
              trainable_new_expert=True,
              prefix_ph='composite/'
              ):
        self.trainable_gate = trainable_gate
        self.trainable_old_expert = trainable_old_expert
        self.trainable_new_expert = trainable_new_expert
        assert isinstance(ob_space, gym.spaces.Box)
        
        self.name = name
        self.independent = independent

        with tf.variable_scope(name):

            ''' Create Placeholders for Observation (Input) '''
            sequence_length = None
            if self.independent:
                assert stochastic is None and ob is None
                ob = U.get_placeholder(name=prefix_ph+'ob', dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
                stochastic = tf.placeholder(name=prefix_ph+'stochastic', dtype=tf.bool, shape=())
            else:
                assert stochastic is not None and ob is not None
            if ob_filter_gate:
                obz, self.ob_rms = NORM_FILTER("obfilter", ob, ob_space)
            else:
                obz = ob

            self.old_experts = []
            self.new_experts = []
            ''' Create existing expert control policies and connect them with observation '''
            ''' The existing experts do not change during learning process '''
            if old_expert_names is not None:
                self.old_experts = [self.expert_fn(n, 
                                                   ob, 
                                                   ob_space, 
                                                   ac_space, 
                                                   hid_size_expert, 
                                                   num_hid_layers_expert, 
                                                   trainable=trainable_old_expert, 
                                                   ob_filter=ob_filter_old_expert, 
                                                   var_reuse=False) for n in old_expert_names]
            ''' Create extra expert control policies which are rooms for improvement for the new task '''
            ''' The new experts will change during learning process '''
            if new_expert_names is not None:
                self.new_experts = [self.expert_fn(n, 
                                                   ob, 
                                                   ob_space, 
                                                   ac_space, 
                                                   hid_size_expert, 
                                                   num_hid_layers_expert, 
                                                   trainable=trainable_new_expert, 
                                                   ob_filter=ob_filter_new_expert, 
                                                   var_reuse=False) for n in new_expert_names]

            self.experts = self.old_experts + self.new_experts
            with tf.variable_scope('pol'):
                self.pdtype = make_pdtype(ac_space)
                
                ''' Create gating network '''
                input_gate = obz
                output_gate, self.scope_gate = \
                    self.gate_fn(input_gate, hid_size_gate, num_hid_layers_gate, trainable_gate)

                self.weights, mean, logstd = self.create_weight_mean_logstd(output_gate)
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)

                self.weights_for_old = self.weights[:,:self.num_old_experts()]
                self.weights_for_new = self.weights[:,self.num_old_experts():]
                
                self.scope_ac = tf.get_variable_scope().name

            ''' Create value network '''
            vpred, self.scope_vf = MLP("vf", 
                                       obz,
                                       1,
                                       hid_size_expert,
                                       num_hid_layers_expert,
                                       input_nonlinearity=False,
                                       output_nonlinearity=False,
                                       trainable=True,
                                       nonlinear_fn=tf.nn.relu)

            self.pd = self.pdtype.pdfromflat(pdparam)
            self.vpred = vpred[:,0]

            ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
            if self.independent:        
                self._act = U.function([stochastic, ob], [ac, self.vpred])
                self._weight = U.function([ob], [self.weights])
                self._weight_for_old = U.function([ob], [self.weights_for_old])
                self._weight_for_new = U.function([ob], [self.weights_for_new])
                # self._act_expert = [U.function([ob], [self.experts[i].output_ac]) for i in range(len(self.experts))]
                # self._val_expert = [U.function([ob], [self.experts[i].output_vf]) for i in range(len(self.experts))]
                self._act_expert, self._val_expert = [], []
                for i in range(len(self.experts)):
                    ac_expert = U.switch(stochastic, self.experts[i].pd.sample(), self.experts[i].pd.mode())
                    vpred_expert = self.experts[i].output_vf
                    self._act_expert.append(U.function([stochastic, ob], [ac_expert, vpred_expert]))
                    self._val_expert.append(U.function([ob], [vpred_expert]))

            self.input = ob
            self.output_ac = self.pd.mode()
            self.output_vf = vpred

            self.scope = tf.get_variable_scope().name

        ''' This allow us to copy nn weights having same shape with different names '''
        self._assign_old_experts = []
        if old_expert_names is not None:
            old_experts_orig = [self.expert_fn("pi/%s"%(n), 
                                                ob, 
                                                ob_space, 
                                                ac_space, 
                                                hid_size_expert, 
                                                num_hid_layers_expert,
                                                trainable=False, 
                                                ob_filter=ob_filter_old_expert, 
                                                var_reuse=True) for n in old_expert_names]
            self.load_experts(old_experts_orig, old_expert_weights)
            self._assign_old_experts = [U.function([],[], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zipsame(self.old_experts[i].get_variables_all(), old_experts_orig[i].get_variables_all())]) for i in range(self.num_old_experts())]
            self._assign_old_experts_orig = [U.function([],[], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zipsame(old_experts_orig[i].get_variables_all(), self.old_experts[i].get_variables_all())]) for i in range(self.num_old_experts())]
            self.old_experts_orig = old_experts_orig
    def gate_fn(self, input, hid_size, num_hid_layers, trainable):
        return MLP("gate", 
                   input,
                   len(self.experts), 
                   hid_size, 
                   num_hid_layers, 
                   input_nonlinearity=False,
                   output_nonlinearity=False,
                   trainable=trainable,
                   nonlinear_fn=tf.nn.relu)
    def expert_fn(self, name, ob, ob_space, ac_space, hid_size, num_hid_layers, trainable, ob_filter, var_reuse):
        return MlpPolicy(name=name,
                         independent=False, 
                         stochastic=False,
                         ob=ob, 
                         ob_space=ob_space, 
                         ac_space=ac_space, 
                         hid_size=hid_size, 
                         num_hid_layers=num_hid_layers,
                         ob_filter=ob_filter, 
                         trainable=trainable, 
                         var_reuse=var_reuse,
                         prefix_ph='expert/')
    def create_weight_mean_logstd(self, output_gate):
        raise NotImplementedError('This should be overrided')
    def act(self, stochastic, ob):
        if self.independent:
            ac1, vpred1 = self._act(stochastic, ob[None])
            return ac1[0], vpred1[0]
        else:
            raise NotImplementedError('dependent mode does not support this')
    def act_expert(self, stochastic, ob, idx=None):
        if idx is None:
            result = []
            for i in range(len(self.experts)):
                ac, vpred = self._act_expert[i](stochastic, ob[None])
                result.append((ac[0], vpred[0]))
            return result
        else:
            ac, vpred = self._act_expert[idx](stochastic, ob[None])
            return ac[0], vpred[0]
    def num_experts(self):
        return len(self.experts)
    def num_new_experts(self):
        return len(self.new_experts)
    def num_old_experts(self):
        return len(self.old_experts)
    def weight(self, ob):
        return self._weight(ob[None])[0][0]
    def weight_for_old(self, ob):
        return self._weight_for_old(ob[None])[0][0]
    def weight_for_new(self, ob):
        return self._weight_for_new(ob[None])[0][0]
    def get_variables_all(self):
        variables = []
        variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
        for i in range(self.num_experts()):
            variables += self.experts[i].get_variables_all()
        return variables
    def get_variables_gate(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_gate)
    def get_variables_new_experts(self, idx=None):
        variables = []
        experts = self.new_experts
        indices = range(len(experts)) if idx is None else [idx]
        for i in indices:
            variables += experts[i].get_variables_all()        
        return variables
    def get_variables_old_experts(self, idx=None):
        variables = []
        experts = self.old_experts
        indices = range(len(experts)) if idx is None else [idx]
        for i in indices:
            variables += experts[i].get_variables_all()        
        return variables
    def get_variables(self, ac=True, vf=True, include_experts=True):
        variables = []
        if ac: variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_ac)
        if vf: variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_vf)
        if include_experts:
            for i in range(self.num_experts()):
                variables += self.experts[i].get_variables(ac, vf)
        return variables
    def get_trainable_variables(self, ac=True, vf=True, include_experts=True):
        variables = []
        if ac: variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_ac)
        if vf: variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_vf)
        if include_experts:
            for i in range(self.num_experts()):
                variables += self.experts[i].get_trainable_variables(ac, vf)
        return variables
    def get_trainable_variables_gate(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_gate)
    def get_trainable_variables_experts(self, idx=None):
        variables = []
        experts = self.experts
        indices = range(len(experts)) if idx is None else [idx]
        for i in indices:
            variables += experts[i].get_trainable_variables()      
        return variables
    def load_experts(self, experts, files):
        assert len(files) == len(experts)
        for i in range(len(files)):
            variables = experts[i].get_variables_all()
            U.load_variables(files[i], variables)
            U.ALREADY_INITIALIZED.update(variables)
    def load_variables(self, file):
        variables = self.get_variables_all()
        U.load_variables(file, variables)
        U.ALREADY_INITIALIZED.update(variables)
    def copy_and_save_experts_orig(self, file, idx):
        assert idx < self.num_old_experts()
        self._assign_old_experts_orig[idx]()
        variables = self.old_experts_orig[idx].get_variables_all()
        print(variables)
        U.save_variables(file, variables)
    def assign_old_experts(self):
        for assign_op in self._assign_old_experts:
            assign_op()

class ValueCompositePolicy(CompositePolicy):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        super(ValueCompositePolicy, self).__init__(name, *args, **kwargs)
    def create_weight_mean_logstd(self, output_gate):
        values = tf.concat([self.experts[i].output_vf for i in range(len(self.experts))], axis=1)
        weights = tf.nn.softmax(values)
        experts_ac_tensor = tf.stack([self.experts[i].output_ac for i in range(len(self.experts))], axis=1)
        weights_tensor = tf.expand_dims(weights, -1)

        mean = tf.reduce_sum(tf.multiply(experts_ac_tensor, weights_tensor), axis=1)
        logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0]//2], trainable=trainable, initializer=tf.zeros_initializer())
        
        return weights, mean, logstd

class AdditivePolicy(CompositePolicy):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        super(AdditivePolicy, self).__init__(name, *args, **kwargs)
    def create_weight_mean_logstd(self, output_gate):
        weights = tf.nn.softmax(output_gate)
        experts_ac_tensor = tf.stack([self.experts[i].pd.mean for i in range(len(self.experts))], axis=1)
        weights_tensor = tf.expand_dims(weights, -1)

        mean = tf.reduce_sum(tf.multiply(experts_ac_tensor, weights_tensor), axis=1)
        logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0]//2], trainable=True, initializer=tf.zeros_initializer())
        
        return weights, mean, logstd

class AdditiveIndPolicy(CompositePolicy):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        super(AdditiveIndPolicy, self).__init__(name, *args, **kwargs)
    def gate_fn(self, input, hid_size, num_hid_layers, trainable):
        return MLP("gate", 
                   input,
                   len(self.experts) * (self.pdtype.param_shape()[0]//2),
                   hid_size,
                   num_hid_layers,
                   input_nonlinearity=False,
                   output_nonlinearity=False,
                   trainable=trainable,
                   nonlinear_fn=tf.nn.relu)
    def create_weight_mean_logstd(self, output_gate):
        weights = tf.reshape(output_gate, (len(self.experts), self.pdtype.param_shape()[0]//2))
        weights = tf.nn.softmax(weights, axis=0)
        experts_ac_tensor = tf.stack([self.experts[i].pd.mean for i in range(len(self.experts))], axis=1)
        weights_tensor = weights

        mean = tf.reduce_sum(tf.multiply(experts_ac_tensor, weights_tensor), axis=1)
        logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0]//2], trainable=True, initializer=tf.zeros_initializer())
        
        return weights, mean, logstd
    def weight(self, ob):
        return self._weight(ob[None])[0]
    def weight_for_old(self, ob):
        return self._weight_for_old(ob[None])[0]
    def weight_for_new(self, ob):
        return self._weight_for_new(ob[None])[0]

class MultiplicativePolicy(CompositePolicy):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        super(MultiplicativePolicy, self).__init__(name, *args, **kwargs)
    def create_weight_mean_logstd(self, output_gate):
        weights = 0.5 * tf.nn.tanh(output_gate) + 0.5
        weights_tensor = tf.expand_dims(weights, -1)

        experts_mean = tf.stack([self.experts[i].pd.mean for i in range(len(self.experts))], axis=1)
        experts_std = tf.stack([self.experts[i].pd.std for i in range(len(self.experts))], axis=1)

        z = weights_tensor / experts_std
        std = 1.0 / tf.reduce_sum(z, axis=1)
        logstd = tf.log(std)
        mean = std * tf.reduce_sum(z * experts_mean, axis=1)
        
        return weights, mean, logstd

class MultiplicativePolicy2(CompositePolicy):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        super(MultiplicativePolicy2, self).__init__(name, *args, **kwargs)
    def create_weight_mean_logstd(self, output_gate):
        weights = tf.nn.softmax(output_gate)
        weights_tensor = tf.expand_dims(weights, -1)

        experts_mean = tf.stack([self.experts[i].pd.mean for i in range(len(self.experts))], axis=1)
        experts_std = tf.stack([self.experts[i].pd.std for i in range(len(self.experts))], axis=1)

        z = weights_tensor / experts_std
        std = 1.0 / tf.reduce_sum(z, axis=1)
        logstd = tf.log(std)
        mean = std * tf.reduce_sum(z * experts_mean, axis=1)
        
        return weights, mean, logstd
    