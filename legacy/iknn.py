import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np

import baselines.common.tf_util as U
from baselines.common import Dataset, fmt_row, cmd_util
from baselines.common.mpi_adam import MpiAdam
from baselines import logger
import policy

from basecode.motion import kinematics_simple as kinematics
from basecode.math import mmMath
from basecode.utils import basics

import pickle
import gzip

to_meters = 5.6444
njoints = 31

motion_files = [
    './data/motion/iknn/LocomotionFlat01_000.bvh',
    './data/motion/iknn/LocomotionFlat02_000.bvh',
    './data/motion/iknn/LocomotionFlat02_001.bvh',
    './data/motion/iknn/LocomotionFlat03_000.bvh',
    './data/motion/iknn/LocomotionFlat04_000.bvh',
    './data/motion/iknn/LocomotionFlat05_000.bvh',
    './data/motion/iknn/LocomotionFlat06_000.bvh',
    './data/motion/iknn/LocomotionFlat06_001.bvh',
    './data/motion/iknn/LocomotionFlat07_000.bvh',
    './data/motion/iknn/LocomotionFlat08_000.bvh',
    './data/motion/iknn/LocomotionFlat08_001.bvh',
    './data/motion/iknn/LocomotionFlat09_000.bvh',
    './data/motion/iknn/LocomotionFlat10_000.bvh',
    './data/motion/iknn/LocomotionFlat11_000.bvh',
    './data/motion/iknn/LocomotionFlat12_000.bvh',

    './data/motion/iknn/LocomotionFlat01_000_mirror.bvh',
    './data/motion/iknn/LocomotionFlat02_000_mirror.bvh',
    './data/motion/iknn/LocomotionFlat02_001_mirror.bvh',
    './data/motion/iknn/LocomotionFlat03_000_mirror.bvh',
    './data/motion/iknn/LocomotionFlat04_000_mirror.bvh',
    './data/motion/iknn/LocomotionFlat05_000_mirror.bvh',
    './data/motion/iknn/LocomotionFlat06_000_mirror.bvh',
    './data/motion/iknn/LocomotionFlat06_001_mirror.bvh',
    './data/motion/iknn/LocomotionFlat07_000_mirror.bvh',
    './data/motion/iknn/LocomotionFlat08_000_mirror.bvh',
    './data/motion/iknn/LocomotionFlat08_001_mirror.bvh',
    './data/motion/iknn/LocomotionFlat09_000_mirror.bvh',
    './data/motion/iknn/LocomotionFlat10_000_mirror.bvh',
    './data/motion/iknn/LocomotionFlat11_000_mirror.bvh',
    './data/motion/iknn/LocomotionFlat12_000_mirror.bvh',

    './data/motion/iknn/WalkingUpSteps01_000.bvh',
    './data/motion/iknn/WalkingUpSteps02_000.bvh',
    './data/motion/iknn/WalkingUpSteps03_000.bvh',
    './data/motion/iknn/WalkingUpSteps04_000.bvh',
    './data/motion/iknn/WalkingUpSteps04_001.bvh',
    './data/motion/iknn/WalkingUpSteps05_000.bvh',
    './data/motion/iknn/WalkingUpSteps06_000.bvh',
    './data/motion/iknn/WalkingUpSteps07_000.bvh',
    './data/motion/iknn/WalkingUpSteps08_000.bvh',
    './data/motion/iknn/WalkingUpSteps09_000.bvh',
    './data/motion/iknn/WalkingUpSteps10_000.bvh',
    './data/motion/iknn/WalkingUpSteps11_000.bvh',
    './data/motion/iknn/WalkingUpSteps12_000.bvh',

    './data/motion/iknn/WalkingUpSteps01_000_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps02_000_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps03_000_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps04_000_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps04_001_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps05_000_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps06_000_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps07_000_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps08_000_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps09_000_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps10_000_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps11_000_mirror.bvh',
    './data/motion/iknn/WalkingUpSteps12_000_mirror.bvh',

    './data/motion/iknn/NewCaptures01_000.bvh',
    './data/motion/iknn/NewCaptures02_000.bvh',
    './data/motion/iknn/NewCaptures03_000.bvh',
    './data/motion/iknn/NewCaptures03_001.bvh',
    './data/motion/iknn/NewCaptures03_002.bvh',
    './data/motion/iknn/NewCaptures04_000.bvh',
    './data/motion/iknn/NewCaptures05_000.bvh',
    './data/motion/iknn/NewCaptures07_000.bvh',
    './data/motion/iknn/NewCaptures08_000.bvh',
    './data/motion/iknn/NewCaptures09_000.bvh',
    './data/motion/iknn/NewCaptures10_000.bvh',
    './data/motion/iknn/NewCaptures11_000.bvh',

    './data/motion/iknn/NewCaptures01_000_mirror.bvh',
    './data/motion/iknn/NewCaptures02_000_mirror.bvh',
    './data/motion/iknn/NewCaptures03_000_mirror.bvh',
    './data/motion/iknn/NewCaptures03_001_mirror.bvh',
    './data/motion/iknn/NewCaptures03_002_mirror.bvh',
    './data/motion/iknn/NewCaptures04_000_mirror.bvh',
    './data/motion/iknn/NewCaptures05_000_mirror.bvh',
    './data/motion/iknn/NewCaptures07_000_mirror.bvh',
    './data/motion/iknn/NewCaptures08_000_mirror.bvh',
    './data/motion/iknn/NewCaptures09_000_mirror.bvh',
    './data/motion/iknn/NewCaptures10_000_mirror.bvh',
    './data/motion/iknn/NewCaptures11_000_mirror.bvh',
]

def MLP(name, input, output_dim, hid_size, num_hid_layers, input_nonlinearity=False, output_nonlinearity=True, trainable=True):
    NONLINEAR_UNIT = tf.nn.relu
    with tf.variable_scope(name):
        output = NONLINEAR_UNIT(input) if input_nonlinearity else input
        for i in range(num_hid_layers):
            output = NONLINEAR_UNIT(tf.layers.dense(output, hid_size, name="fc%i"%(i+1), trainable=trainable, kernel_initializer=U.normc_initializer(1.0)))
        output = tf.layers.dense(output, output_dim, name="final", trainable=trainable, kernel_initializer=U.normc_initializer(1.0))
        output = NONLINEAR_UNIT(output) if output_nonlinearity else output
        scope = tf.get_variable_scope().name
    return output, scope

class MODEL(object):
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
    def _init(self, dim_in=None, dim_out=None, hid_size=512, num_hid_layers=2):
        if dim_in is None:
            dim_in = 3 * (njoints-1)
        if dim_out is None:
            dim_out = 3 * (njoints-1)
        x = U.get_placeholder(name="x", dtype=tf.float32, shape=[None, dim_in])
        y, _ = MLP("Net", x, dim_out, hid_size, num_hid_layers, False, False, trainable=True)
        self.y = y
        self.x = x
        self._eval = U.function([x], [y])
    def load(self, file):
        variables = self.get_variables_all()
        U.load_variables(file, variables)
        U.ALREADY_INITIALIZED.update(variables)
    def eval(self, x):
        return self._eval(x)[0][0]
    def get_variables_all(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

def process_data(motion_files):
    X = []
    Y = []
    skel = None

    for file in motion_files:
        pre, ext = os.path.splitext(file)
        m = None
        if ext == ".bin":
            m = pickle.load(open(pre+".bin", 'rb'))
            print(pre+".bin")
        elif os.path.exists(pre+".bin"):
            m = pickle.load(open(pre+".bin", 'rb'))
            print(pre+".bin")
        elif os.path.exists(pre+".gzip"):
            with gzip.open(pre+".bin", "rb") as f:
                m = pickle.load(f)
            print(pre+".gzip")
        elif ext == ".bvh":
            m = kinematics.Motion(file=file, scale=to_meters, vup='y')
            with gzip.open(pre+".gzip", "wb") as f:
                pickle.dump(m, f)
            print(file)
        else:
            continue
        
        num_pose = 0
        for pose in m.postures:
            skel = pose.skel
            root_joint = skel.root_joint
            T0 = pose.get_transform(root_joint, local=False)
            T0_inv = mmMath.invertSE3(T0)
            x = [mmMath.T2p(np.dot(T0_inv, pose.get_transform(j, local=False))) for j in range(njoints)]
            y = [mmMath.logSO3(mmMath.T2R(pose.get_transform(j, local=True))) for j in range(njoints)]
            # print(mmMath.T2p(np.dot(T0_inv, pose.get_transform(30, local=False))))
            X.append(np.hstack(x))
            Y.append(np.hstack(y))
            num_pose += 1
            if num_pose % 100: print('\r%d posetures are processed' % (num_pose), end=" ")
        print(" ")

    assert len(X) > 0 and len(Y) > 0

    print('==========Total %d postures are process==========' % len(X))

    # joint_parents = []
    # for m in motions:
    #     for pose in m.postures:
    #         for joint in pose.skel.joints:
    #             if joint.parent_joint is None:
    #                 joint_parents.append(-1)
    #             else:
    #                 joint_parents.append(pose.skel.get_joint_index(joint.parent_joint))
    #         break
    #     break

    return np.array(X), np.array(Y), skel

def train(model, X_train, Y_train, X_test=None, Y_test=None, rec_name=None):
    assert len(X_train)==len(Y_train)
    assert rec_name is not None
    
    d_train = Dataset(dict(X=X_train, Y=Y_train), shuffle=True)
    d_test = None

    dim_in, dim_out = len(X_train[0]), len(Y_train[0])

    y_target = tf.placeholder(dtype=tf.float32, shape=[None, dim_out])
    loss = tf.reduce_mean(tf.square(model.y - y_target))
    var_list = model.get_variables_all()
    compute_lossandgrad = U.function([model.x, y_target], [loss, U.flatgrad(loss, var_list, None)])
    compute_loss = U.function([model.x, y_target], [loss])

    if X_test is not None and Y_test is not None:
        assert len(X_test)==len(Y_test)
        d_test = Dataset(dict(X=X_test, Y=Y_test), shuffle=True)

    adam = MpiAdam(var_list, epsilon=1e-5)

    U.initialize()
    adam.sync()

    opt_iter = 10000
    opt_batchsize = 256
    opt_stepsize = 1.0e-04
    log_data = [0, 0, 0]
    logger.log(fmt_row(13, ['Iteration', 'LossTrain', 'LossTest']))
    for i in range(opt_iter):
        log_data[0] = i
        cur_lrmult =  max(1.0-float(i)/opt_iter, 0)
        losses_train = []
        for batch in d_train.iterate_once(opt_batchsize):
            args = [batch['X'], batch['Y']]
            *newlosses, g = compute_lossandgrad(*args)
            adam.update(g, opt_stepsize*cur_lrmult)
            losses_train.append(newlosses)
        log_data[1] = np.mean(losses_train)
        if d_test is not None:
            losses_test = []
            for batch in d_test.iterate_once(len(X_test)):
                args = [batch['X'], batch['Y']]
                newlosses = compute_loss(*args)
                losses_test.append(newlosses)
            log_data[2] = np.mean(losses_test)
        logger.log(fmt_row(13, log_data))

        if i%50 == 0:
            U.save_variables(rec_name+"_"+str(i), model.get_variables_all())

def keyboard_callback(key):
    global model, X_test, Y_test
    global x_cur, y_cur
    global input_positions_global, output_positions_global, skel
    
    if key == b' ':
        idx = int(np.random.uniform(0, len(X_test)))
        x = X_test[idx,3:]
        y = Y_test[idx,3:]
        y_pred = model.eval(x)

        x = np.reshape(x, (-1, 3))
        y_pred = np.reshape(y_pred, (-1, 3))
        
        for i in range(njoints-1):
            input_positions_global[i+1] = x[i]

        for i in range(njoints-1):
            joint = skel.joints[i+1]
            T_rest = joint.info['tf_base_local']
            T_move = mmMath.R2T(mmMath.exp(y_pred[i]))
            T = np.dot(T_rest, T_move)
            while joint.parent_joint is not None:
                T_rest = joint.parent_joint.info['tf_base_local']
                T_move = mmMath.R2T(mmMath.exp(y_pred[skel.get_joint_index(joint.parent_joint)-1]))
                T = np.dot(np.dot(T_rest, T_move), T)
                joint = joint.parent_joint
            output_positions_global[i+1] = mmMath.T2p(T)
    elif key == b'\x1b':
        exit(0)
    else:
        raise NotImplementedError('key:'+str(key))

def render_callback():
    global input_positions_global, output_positions_global, skel

    gl_render.render_ground(size=[100, 100], color=[0.8, 0.8, 0.8], axis='y', origin=True, use_arrow=True)

    glEnable(GL_LIGHTING)

    glPushMatrix()
    glScalef(0.01, 0.01, 0.01)

    # for joint in skel.joints:
    #     p1 = mmMath.T2p(joint.info['tf_base_global'])
    #     gl_render.render_point(p=p1, radius=0.02, color=[0.8, 0, 0, 1])
    #     if joint.parent_joint is not None:
    #         p2 = mmMath.T2p(joint.parent_joint.info['tf_base_global'])
    #         gl_render.render_line(p1=p1, p2=p2, color=[0, 0, 0, 1])

    for i in range(njoints):
        p1 = input_positions_global[i]
        gl_render.render_point(p=p1, radius=2.0, color=[0, 0.8, 0, 1])
        joint = skel.joints[i]
        if joint.parent_joint is not None:
            ii = skel.get_joint_index(joint.parent_joint)
            p2 = input_positions_global[ii]
            gl_render.render_line(p1=p1, p2=p2, color=[0, 0.5, 0, 1])

    for i in range(njoints):
        p1 = output_positions_global[i]
        gl_render.render_point(p=p1, radius=2.0, color=[0.8, 0, 0, 1])
        joint = skel.joints[i]
        if joint.parent_joint is not None:
            ii = skel.get_joint_index(joint.parent_joint)
            p2 = output_positions_global[ii]
            gl_render.render_line(p1=p1, p2=p2, color=[0.5, 0, 0, 1])

    glPopMatrix()

def arg_parser():
    parser = cmd_util.arg_parser()
    parser.add_argument('--num-iterations', type=int, default=int(1000))
    parser.add_argument('--mode', help='MODE [train, load]', type=str, default="train")
    parser.add_argument('--model', help='Loading Network', type=str, default="iknn")
    parser.add_argument('--dir', help='Loading Network', type=str, default="data/learning/iknn/")
    return parser

if __name__ == '__main__':

    args = arg_parser().parse_args()
    logger.configure()
    os.makedirs(args.dir, exist_ok = True)

    print('================================================')
    print('========Inverse Kinematics by Neural Net========')
    print('================================================')
    X, Y, skel = process_data(motion_files)
    print('Numdata:', len(X), len(Y))
    print('Dimension:', len(X[0]), len(Y[0]))

    num_data = len(X)
    num_data_train = int(0.9*num_data)
    X_train, Y_train = X[:num_data_train, :], Y[:num_data_train, :]
    X_test, Y_test = X[num_data_train:, :], Y[num_data_train:, :]

    dim_in = len(X_test[0])-3
    dim_out = len(Y_test[0])-3

    model = MODEL('IKNN', dim_in, dim_out)

    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *

    from basecode.render import gl_render
    from basecode.render import glut_viewer as viewer

    if args.mode == "train":
        train(model, X_train[:,3:], Y_train[:,3:], X_test[:,3:], Y_test[:,3:], rec_name=args.dir+args.model)
    elif args.mode == "load":
        model.load(args.dir+args.model)
    else:
        raise NotImplementedError

    cam_origin = np.zeros(3)
    cam_pos = cam_origin + np.array([0.0, 1.0, 3.5])

    input_positions_global = [np.zeros(3) for i in range(njoints)]
    output_positions_global = [np.zeros(3) for i in range(njoints)]

    viewer.run(
        title="Inverse Kinematics by Neural Net",
        cam_pos=cam_pos,
        cam_origin=cam_origin,
        size=(1280, 720),
        keyboard_callback=keyboard_callback,
        render_callback=render_callback,
        )