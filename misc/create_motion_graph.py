'''
TODO
parallel read_motions does not work when using 'bvh'
'''

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import os
import sys
import copy
import numpy as np
import importlib.util

from basecode.motion import kinematics_simple as kinematics
from basecode.motion import motion_graph as mg
from basecode.math import mmMath
from basecode.utils import multiprocessing as mp
from basecode.utils import basics

from baselines.common import cmd_util
import matplotlib.pyplot as plt

import networkx as nx
import pickle
import gzip
import random

def create_random_motions(job_idx, filename, max_length, min_length):
    np.random.seed(os.getpid())
    random.seed(os.getpid())
    res = []
    if job_idx[0] >= job_idx[1]:
        return res
    mg = mp.shared_data
    idx = job_idx[0]
    total_cnt = job_idx[1] - job_idx[0]
    cnt = 0
    time_checker = basics.TimeChecker()
    while cnt < total_cnt:
        m = mg.create_random_motion(max_length, start_idx=None)
        if m.times[-1] < min_length:
            continue
        with gzip.open("%s_%04d.gzip"%(filename,idx+cnt), "wb") as f:
            pickle.dump(m, f)
        cnt += 1
        if cnt % int(0.1*total_cnt)==0:
            percentage = 100*cnt/float(total_cnt)
            time_elapsed = time_checker.get_time(restart=True)
            print('[%d] %f %% completed / %f sec'%(mp.get_pid(),percentage,time_elapsed))
    return res

def read_motions(job_idx, skel, scale, vup):
    res = []
    if job_idx[0] >= job_idx[1]:
        return res
    for i in range(job_idx[0], job_idx[1]):
        file = mp.shared_data[i]
        pre, ext = os.path.splitext(file)
        if file.endswith('bvh'):
            motion = kinematics.Motion(skel=skel,
                                       file=file,
                                       scale=scale,
                                       load_skel=False,
                                       vup=vup)
            motion.resample(fps=24)
            with gzip.open(pre+".gzip", "wb") as f:
                pickle.dump(motion, f)
        elif file.endswith('bin'):
            motion = pickle.load(open(file, "rb"))
        elif file.endswith('gzip'):
            with gzip.open(file, "rb") as f:
                motion = pickle.load(f)
        else:
            raise Exception('Unknown Motion File Type')
        res.append(motion)
        print('Loaded: %s'%file)
    return res

def keyboard_callback(key):
    global viewer, motion, cur_time, w
    global time_checker_auto_play

    if key in toggle:
        flag[toggle[key]] = not flag[toggle[key]]
        print('Toggle:', toggle[key], flag[toggle[key]])
    elif key == b'r':
        cur_time = 0.0
        time_checker_auto_play.begin()
        time_checker_global.begin()
    elif key == b'R':
        motion = motion_graph.create_random_motion(length=20.0)
        cur_time = 0.0
        time_checker_auto_play.begin()
        time_checker_global.begin()
    elif key == b']':
        # print('---------')
        # print(cur_time, motion.time_to_frame(cur_time))
        pose1 = motion.get_pose_by_frame(motion.time_to_frame(cur_time))
        vel1 = motion.get_velocity_by_frame(motion.time_to_frame(cur_time))
        next_frame = min(motion.num_frame()-1,
                         motion.time_to_frame(cur_time)+1)
        cur_time = motion.frame_to_time(next_frame)
        pose2 = motion.get_pose_by_frame(motion.time_to_frame(cur_time))
        vel2 = motion.get_velocity_by_frame(motion.time_to_frame(cur_time))
        # print(cur_time, next_frame)

        diff_pose = kinematics.pose_similiarity(pose1,
                                                pose2,
                                                vel1,
                                                vel2,
                                                w['w_joint_pos'],
                                                w['w_joint_vel'],
                                                w['w_joints'])
        diff_root_ee = kinematics.root_ee_similarity(pose1,
                                                     pose2,
                                                     vel1,
                                                     vel2,
                                                     w['w_root_pos'],
                                                     w['w_root_vel'],
                                                     w['w_ee_pos'],
                                                     w['w_ee_vel'])
        diff = diff_pose + diff_root_ee
        print('diff: ', diff, ' / ', diff_pose, diff_root_ee)
    elif key == b'[':
        prev_frame = max(0,
                         motion.time_to_frame(cur_time)-1)
        cur_time = motion.frame_to_time(prev_frame)
    else:
        return False
    return True

def idle_callback():
    global viewer, flag, motion, cur_time
    global time_checker_auto_play

    time_elapsed = time_checker_auto_play.get_time(restart=False)
    if flag['auto_play']:
        cur_time += time_elapsed
    time_checker_auto_play.begin()

    if flag['follow_cam']:
        pass
        # p, _, _, _ = env._sim_agent.get_root_state()
        # if env._yup:
        #     viewer.update_target_pos(p, ignore_y=True)
        # else:
        #     viewer.update_target_pos(p, ignore_z=True)

def render_callback():
    global motion, flag, cur_time

    skel = motion.skel

    if flag['ground']:
        gl_render.render_ground(size=[100, 100], color=[0.9, 0.9, 0.9], axis=skel.vup, origin=flag['origin'], use_arrow=True)

    pose = motion.get_pose_by_frame(motion.time_to_frame(cur_time))
    glEnable(GL_LIGHTING)
    for j in skel.joints:
        pos = mmMath.T2p(pose.get_transform(j, local=False))
        gl_render.render_point(pos, radius=0.03, color=[0.8, 0.8, 0.0, 1.0])
        if j.parent_joint is not None:
            pos_parent = mmMath.T2p(pose.get_transform(j.parent_joint, local=False))
            gl_render.render_line(p1=pos_parent, p2=pos, color=[0.5, 0.5, 0, 1])

def overlay_callback():
    return

def arg_parser():
    parser = cmd_util.arg_parser()
    parser.add_argument('--motion_files', action='append', help='Motion Files')
    parser.add_argument('--vup', type=str, default='y')
    parser.add_argument('--scale', type=float, default=0.056444)
    parser.add_argument('--stride', type=float, default=1.5)
    parser.add_argument('--blend_window', type=float, default=0.2)
    parser.add_argument('--diff_threshold', type=float, default=0.7)
    parser.add_argument('--w_joint_root_scale', type=float, default=2.0)
    parser.add_argument('--w_joint_pos', type=float, default=0.4)
    parser.add_argument('--w_joint_vel', type=float, default=0.1)
    parser.add_argument('--w_root_pos', type=float, default=0.7)
    parser.add_argument('--w_root_vel', type=float, default=0.3)
    parser.add_argument('--w_ee_pos', type=float, default=0.7)
    parser.add_argument('--w_ee_vel', type=float, default=0.3)
    parser.add_argument('--char_info_module', type=str, default="mimicpfnn_char_info.py")
    parser.add_argument('--base_motion_file', type=str, default="data/motion/pfnn/pfnn_hierarchy.bvh")
    parser.add_argument('--save_graph_file', type=str, default=None)
    parser.add_argument('--load_graph_file', type=str, default=None)
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument('--create_motion', type=str, default="false")
    parser.add_argument('--create_motion_num_motions', type=int, default=64)
    parser.add_argument('--create_motion_filename', type=str, default="data/temp/motions_by_mg")
    parser.add_argument('--create_motion_max_length', type=float, default=60.0)
    parser.add_argument('--create_motion_min_length', type=float, default=10.0)
    return parser

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        words = None
        with open(sys.argv[1], 'rb') as f:
            words = [word.decode() for line in f for word in line.split()]
            f.close()
        if words is None:
            raise Exception('Invalid Argument')
        args = arg_parser().parse_args(words)
    else:
        args = arg_parser().parse_args()

    spec = importlib.util.spec_from_file_location("char_info", args.char_info_module)
    char_info = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(char_info)
    char_info = char_info

    skel = kinematics.Motion(file=args.base_motion_file, scale=0.01, load_motion=False).skel

    joint_weights = []
    for j in skel.joints:
        idx = char_info.bvh_map_inv[j.name]
        if idx is None:
            joint_weights.append(0.0)
        else:
            w = char_info.joint_weight[idx]
            if j==skel.root_joint:
                w *= args.w_joint_root_scale
            joint_weights.append(w)
    joint_weights = np.array(joint_weights)

    w = {}
    w['w_joints'] = joint_weights
    w['w_joint_pos'] = args.w_joint_pos
    w['w_joint_vel'] = args.w_joint_vel
    w['w_root_pos'] = args.w_root_pos
    w['w_root_vel'] = args.w_root_vel
    w['w_ee_pos'] = args.w_ee_pos
    w['w_ee_vel'] = args.w_ee_vel
    
    ''' Load Motions '''
    if args.load_graph_file is None:
        print(args.motion_files)
        motion_files = args.motion_files
        motions = []
        if isinstance(motion_files[0], str):
            mp.shared_data = motion_files
            motions = mp.run_parallel_async_idx(read_motions, 
                                                args.num_worker, 
                                                len(mp.shared_data), 
                                                skel, 
                                                args.scale, 
                                                args.vup)
            total_length = 0.0
            for m in motions:
                total_length += m.times[-1]
            print('Total %.3f sec long motions were loaded'%total_length)
            print('FPS ~ ', motions[0].num_frame()/motions[0].times[-1])
        else:
            raise Exception('Unknown Type for Reference Motion')
        ''' Construct Motion Graph '''
        motion_graph = mg.MotionGraph(verbose=True)
        motion_graph.auto_construct(motions,
                                    stride=args.stride,
                                    blend_window=args.blend_window,
                                    w_joints=joint_weights,
                                    w_joint_pos=args.w_joint_pos,
                                    w_joint_vel=args.w_joint_vel,
                                    w_root_pos=args.w_root_pos,
                                    w_root_vel=args.w_root_vel,
                                    w_ee_pos=args.w_ee_pos,
                                    w_ee_vel=args.w_ee_vel,
                                    diff_threshold=args.diff_threshold,
                                    num_worker=args.num_worker,
                                    )
        print('Save motion graph to a binary file ...')
        if args.save_graph_file is not None:
            file = args.save_graph_file
        else:
            file = 'data/temp/motion_graph_temp.gzip'
        with gzip.open(file, "wb") as f:
            pickle.dump(motion_graph, f)
    else:
        print('Load motion graph by a binary file ...')
        if args.load_graph_file is not None:
            with gzip.open(args.load_graph_file, "rb") as f:
                motion_graph = pickle.load(f)

    if args.create_motion=="true":
        motion_graph.verbose=False
        mp.shared_data = motion_graph
        res = mp.run_parallel_async_idx(create_random_motions, 
                                        args.num_worker, 
                                        args.create_motion_num_motions,
                                        args.create_motion_filename,
                                        args.create_motion_max_length, 
                                        args.create_motion_min_length,
                                        )
        exit(0)
    
    nx.draw(motion_graph.graph, with_labels=True)
    plt.show()

    motion = motion_graph.create_random_motion(length=20.0, start_idx=0)
    cur_time = 0.0

    from basecode.render import glut_viewer as viewer
    from basecode.render import gl_render
    from basecode.utils import basics

    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *

    # For viewers
    flag = {}
    flag['follow_cam'] = True
    flag['ground'] = True
    flag['origin'] = True
    flag['auto_play'] = False

    toggle = {}
    toggle[b'0'] = 'follow_cam'
    toggle[b'1'] = 'ground'
    toggle[b'2'] = 'origin'
    toggle[b'a'] = 'auto_play'

    time_checker_auto_play = basics.TimeChecker()
    time_checker_global = basics.TimeChecker()

    print('===== Motion Graph =====')

    cam_origin = mmMath.T2p(motion.get_pose_by_time(0.0).get_root_transform())
    if motion.skel.vup=='x':
        cam_pos = cam_origin + np.array([2.0, -3.0, 0.0])
        cam_vup = np.array([1.0, 0.0, 0.0])
    elif motion.skel.vup=='y':
        cam_pos = cam_origin + np.array([0.0, 2.0, -3.0])
        cam_vup = np.array([0.0, 1.0, 0.0])
    elif motion.skel.vup=='z':
        cam_pos = cam_origin + np.array([-3.0, 0.0, 2.0])
        cam_vup = np.array([0.0, 0.0, 1.0])
    else:
        raise NotImplementedError

    viewer.run(
        title='Motion Graph Viewer',
        cam_pos=cam_pos,
        cam_origin=cam_origin,
        cam_vup=cam_vup,
        size=(1280, 720),
        keyboard_callback=keyboard_callback,
        render_callback=render_callback,
        overlay_callback=overlay_callback,
        idle_callback=idle_callback,
        )

