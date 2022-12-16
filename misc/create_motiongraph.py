'''
TODO
'''

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

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

import argparse

import networkx as nx
import pickle
import gzip
import random
import re
import gc

def create_random_motions(job_idx, filename_prefix, max_length, min_length):
    np.random.seed(os.getpid())
    random.seed(os.getpid())
    visited_nodes_all = []
    num_job = job_idx[1] - job_idx[0]
    if num_job <= 0: return res
    motion_graph = mp.shared_data
    idx = job_idx[0]
    cnt = 0
    num_checkpoint = 10
    cnt_checkpoint = int(1.0/num_checkpoint*num_job)
    time_checker = basics.TimeChecker()
    while cnt < num_job:
        m, visited_nodes = motion_graph.create_random_motion(max_length)
        if m.times[-1] < min_length: continue
        m.save_bvh("%s_%04d.bvh"%(filename_prefix,idx+cnt), verbose=False)
        visited_nodes_all.append(visited_nodes)
        cnt += 1
        if  cnt_checkpoint > 0 and cnt % cnt_checkpoint==0:
            percentage = num_checkpoint*(cnt//cnt_checkpoint)
            time_elapsed = time_checker.get_time(restart=True)
            print('[%d] %.2f %% completed / %f sec'%(mp.get_pid(),percentage,time_elapsed))
    return visited_nodes_all

def read_motions(job_idx, skel, scale, v_up_skel, v_face_skel, v_up_env):
    res = []
    if job_idx[0] >= job_idx[1]:
        return res
    for i in range(job_idx[0], job_idx[1]):
        file = mp.shared_data[i]
        if file.endswith('bvh'):
            motion = kinematics.Motion(skel=skel,
                                       file=file,
                                       scale=scale,
                                       load_skel=False,
                                       v_up_skel=v_up_skel, 
                                       v_face_skel=v_face_skel,
                                       v_up_env=v_up_env)
            motion.resample(fps=30)
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
    global viewer, motion, motion_raw, cur_time, args, joint_weights
    global time_checker_auto_play

    if key in toggle:
        flag[toggle[key]] = not flag[toggle[key]]
        print('Toggle:', toggle[key], flag[toggle[key]])
    elif key == b'r':
        cur_time = 0.0
        time_checker_auto_play.begin()
        time_checker_global.begin()
    elif key == b'R':
        print('Reset')
        motion, motion_raw = motion_graph._create_random_motion_debug(length=10.0)
        cur_time = 0.0
        time_checker_auto_play.begin()
        time_checker_global.begin()
    elif key == b']':
        # print('---------')
        # print(cur_time, motion.time_to_frame(cur_time))
        cur_frame = motion.time_to_frame(cur_time)
        next_frame = min(motion.num_frame()-1, cur_frame+1)
        pose1 = motion.get_pose_by_frame(cur_frame)
        vel1 = motion.get_velocity_by_frame(cur_frame)
        pose2 = motion.get_pose_by_frame(next_frame)
        vel2 = motion.get_velocity_by_frame(next_frame)
        # print(cur_time, next_frame)

        # diff_pose = kinematics.pose_similiarity(pose1,
        #                                         pose2,
        #                                         vel1,
        #                                         vel2,
        #                                         args.w_joint_pos,
        #                                         args.w_joint_vel,
        #                                         joint_weights)
        # diff_root_ee = kinematics.root_ee_similarity(pose1,
        #                                              pose2,
        #                                              vel1,
        #                                              vel2,
        #                                              args.w_root_pos,
        #                                              args.w_root_vel,
        #                                              args.w_ee_pos,
        #                                              args.w_ee_vel)
        # diff = diff_pose + diff_root_ee
        # print('diff: ', diff, ' / ', diff_pose, diff_root_ee)

        cur_time = motion.frame_to_time(next_frame)
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
    global motion, flag, cur_time, char_info

    skel = motion.skel

    if flag['ground']:
        gl_render.render_ground(size=[100, 100], 
                                color=[0.9, 0.9, 0.9], 
                                axis=kinematics.axis_to_str(char_info.v_up_env),
                                origin=flag['origin'], 
                                use_arrow=True)
    def render_pose(pose, color_joint, color_link):
        glEnable(GL_LIGHTING)
        for j in skel.joints:
            pos = mmMath.T2p(pose.get_transform(j, local=False))
            gl_render.render_point(pos, radius=0.01, color=color_joint)
            if j.parent_joint is not None:
                pos_parent = mmMath.T2p(pose.get_transform(j.parent_joint, local=False))
                gl_render.render_line(p1=pos_parent, p2=pos, color=color_link)
    cur_frame = motion.time_to_frame(cur_time)
    num_draw = 5
    for i in range(num_draw):
        frame = min(cur_frame+i, motion.num_frame()-1)
        alpha = 1.0 - (i+1.0)/(num_draw)
        render_pose(motion.get_pose_by_frame(frame),
                    color_joint=[0.8, 0.8, 0, alpha],
                    color_link=[0.5, 0.5, 0.5, alpha])
        render_pose(motion_raw.get_pose_by_frame(frame), 
                    color_joint=[0.8, 0, 0, alpha],
                    color_link=[0.7, 0.7, 0.7, alpha])

def overlay_callback():
    global viewer, motion, cur_time
    glPushAttrib(GL_LIGHTING)
    glDisable(GL_LIGHTING)
    
    w, h = viewer.window_size
    ''' FPS '''
    gl_render.render_text("FPS: %.2f"%viewer.avg_fps, pos=[0.05*w, 0.9*h], font=GLUT_BITMAP_9_BY_15)
    gl_render.render_text("Time: %.2f / %.2f"%(cur_time, motion.times[-1]), pos=[0.05*w, 0.9*h+20], font=GLUT_BITMAP_9_BY_15)
    glPopAttrib(GL_LIGHTING)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='[create, load, merge]', default='create')
    parser.add_argument('--visualize', type=str, default="false")
    parser.add_argument('--motion_files', action='append', help='Motion Files')
    parser.add_argument('--motion_dir', type=str, help='Motion Directory')
    parser.add_argument('--motion_labels', type=str, help='Motion Labels')
    parser.add_argument('--scale_motion', type=float, default=1.0)
    parser.add_argument('--scale_skel', type=float, default=1.0)
    parser.add_argument('--base_length', type=float, default=1.5)
    parser.add_argument('--stride_length', type=float, default=1.5)
    parser.add_argument('--blend_length', type=float, default=0.5)
    parser.add_argument('--compare_length', type=float, default=1.0)
    parser.add_argument('--num_comparison', type=int, default=3)
    parser.add_argument('--diff_threshold', type=float, default=0.5)
    parser.add_argument('--w_joint_root_scale', type=float, default=2.0)
    parser.add_argument('--w_joint_pos', type=float, default=8.0)
    parser.add_argument('--w_joint_vel', type=float, default=2.0)
    parser.add_argument('--w_root_pos', type=float, default=0.5)
    parser.add_argument('--w_root_vel', type=float, default=1.5)
    parser.add_argument('--w_ee_pos', type=float, default=3.0)
    parser.add_argument('--w_ee_vel', type=float, default=1.0)
    parser.add_argument('--w_trajectory', type=float, default=1.0)
    parser.add_argument('--char_info_module', type=str, default=None)
    parser.add_argument('--base_motion_file', type=str, default=None)
    parser.add_argument('--save_graph_file', type=str, default=None)
    parser.add_argument('--load_graph_file', type=str, default=None)
    parser.add_argument('--graph_reduce', action='append', type=str, default=[])
    parser.add_argument('--graph_reduce_threshold', type=float, default=1.0)
    parser.add_argument('--graph_reduce_num_component', type=int, default=1)
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--create_motion', type=str, default="false")
    parser.add_argument('--create_motion_num_motions', type=int, default=64)
    parser.add_argument('--create_motion_filename', type=str, default="data/temp/motions_by_mg")
    parser.add_argument('--create_motion_max_length', type=float, default=60.0)
    parser.add_argument('--create_motion_min_length', type=float, default=10.0)
    parser.add_argument('--analysis', type=str, default="false")
    parser.add_argument('--num_machine', type=int, default=1)
    parser.add_argument('--id_machine', type=int, default=0)
    parser.add_argument('--merge_graph', type=str, default="false")
    parser.add_argument('--merge_dir', type=str, help='Motion Graph Directory')
    parser.add_argument('--load_motions_in_advance', type=str, default='false')
    parser.add_argument('--verbose', type=str, default='true')
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

    skel = kinematics.Motion(file=args.base_motion_file, 
                             scale=args.scale_skel, 
                             load_motion=False,
                             v_up_skel=char_info.v_up, 
                             v_face_skel=char_info.v_face,
                             v_up_env=char_info.v_up_env).skel

    ''' Get joint weight values form char_info '''
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

    def get_motion_files_with_label(motion_dir, labels=None, ext=".bvh"):
        motion_files = []
        files = basics.files_in_dir(motion_dir, ext)
        if labels is not None:
            keys_labels = []
            with open(labels, 'r') as file:
                for line in file:
                    l = re.split('[\t|\n| ]+', line)
                    key = '%s/%s'%(l[0],l[1])
                    label = False if l[2] =='0' else True
                    keys_labels.append((key, label))
            files_filtered = []
            for key, label in keys_labels:
                for file in files:
                    if key in file: 
                        if label: files_filtered.append(file)
            files = files_filtered
        motion_files += files
        return motion_files

    if args.mode == 'merge_graph':
        ''' 
        This merges the existing motion graphs, which are constructed 
        by different machine respectively. This assumes that all graphs
        have same nodes, they could have different edges.
        '''
        assert args.merge_dir is not None
        graphs = []
        files = basics.files_in_dir(args.merge_dir, ".mg.graph.gzip")
        for file in files:
            gc.collect()
            with gzip.open(file, "rb") as f:
                g = pickle.load(f)
                graphs.append(g)
                nn = g.number_of_nodes()
                ne = g.number_of_edges()
                print('Loaded: %s (%d nodes / %d edges)'%(file, nn, ne))
        if len(graphs)==0:
            print('No motion graphs in the directory!!!')
            exit(0)
        ''' Save network only '''
        graph_merged = nx.compose_all(graphs)
        filename = "%s/merged.mg.graph.gzip"%args.merge_dir
        with gzip.open(filename, "wb") as f:
            pickle.dump(graph_merged, f)
            nn = graph_merged.number_of_nodes()
            ne = graph_merged.number_of_edges()
            print('Saved: %s (%d nodes / %d edges)'%(filename, nn, ne))        
        exit(0)
    elif args.mode == 'load':
        assert args.load_graph_file is not None
        assert args.motion_dir is not None
        print('Load motion graph by a binary file ...')
        motion_files = get_motion_files_with_label(args.motion_dir, labels=args.motion_labels, ext=".bvh")
        motion_graph = mg.MotionGraph(motion_files=motion_files,
                                      skel=skel,
                                      fps=args.fps, 
                                      base_length=args.base_length,
                                      stride_length=args.stride_length,
                                      blend_length=args.blend_length,
                                      compare_length=args.compare_length,
                                      verbose=args.verbose=='true')
        motion_graph.load_graph_only(args.load_graph_file)
        if args.graph_reduce is not None:
            graph = motion_graph.graph
            if 'edge_weight' in args.graph_reduce:
                nn1 = graph.number_of_nodes()
                ne1 = graph.number_of_edges()
                args.diff_threshold
                deleted_edges = []
                for u, v in graph.edges:
                    data = graph.get_edge_data(u, v)
                    if data['weights'] > args.graph_reduce_threshold:
                        deleted_edges.append((u, v))
                graph.remove_edges_from(deleted_edges)
                nn2 = graph.number_of_nodes()
                ne2 = graph.number_of_edges()
                print('The graph was reduced by EdgeThreshold %f'%args.graph_reduce_threshold)
                print('nodes (%d -> %d) / edges (%d -> %d)'%(nn1,nn2,ne1,ne2))
            if 'wcc' in args.graph_reduce:
                nn1 = graph.number_of_nodes()
                ne1 = graph.number_of_edges()
                motion_graph.reduce('wcc', args.graph_reduce_num_component)
                nn2 = graph.number_of_nodes()
                ne2 = graph.number_of_edges()
                print('The graph was reduced by wcc')
                print('nodes (%d -> %d) / edges (%d -> %d)'%(nn1,nn2,ne1,ne2))
            if 'scc' in args.graph_reduce:
                nn1 = graph.number_of_nodes()
                ne1 = graph.number_of_edges()
                motion_graph.reduce('scc', args.graph_reduce_num_component)
                nn2 = graph.number_of_nodes()
                ne2 = graph.number_of_edges()
                print('The graph was reduced by scc')
                print('nodes (%d -> %d) / edges (%d -> %d)'%(nn1,nn2,ne1,ne2))
        
        # nodes = [10124, 10150, 11993, 12011, 7004, 4748, 4763, 1488, 15392, 15406]
        # m = motion_graph.create_motion_by_following(nodes)
        # m.save_bvh('data/temp/test.bvh')

        # m, visited_nodes = motion_graph.create_random_motion(10)
        # print(visited_nodes)
        # exit(0)
        
        if args.load_motions_in_advance == 'true':
            motion_graph.load_motions(num_worker=args.num_worker, scale=args.scale_motion)
    elif args.mode == 'create':
        if args.motion_files is None:
            motion_files = []
        else:
            motion_files = args.motion_files
        motion_files += get_motion_files_with_label(args.motion_dir, labels=args.motion_labels, ext=".bvh")
        motion_graph = mg.MotionGraph(motion_files=motion_files,
                                      skel=skel,
                                      fps=args.fps, 
                                      base_length=args.base_length,
                                      stride_length=args.stride_length,
                                      blend_length=args.blend_length,
                                      compare_length=args.compare_length,
                                      verbose=args.verbose=='true')
        motion_graph.load_motions(num_worker=args.num_worker, scale=args.scale_motion)
        motion_graph.construct(w_joints=joint_weights,
                               w_joint_pos=args.w_joint_pos,
                               w_joint_vel=args.w_joint_vel,
                               w_root_pos=args.w_root_pos,
                               w_root_vel=args.w_root_vel,
                               w_ee_pos=args.w_ee_pos,
                               w_ee_vel=args.w_ee_vel,
                               w_trajectory=args.w_trajectory,
                               diff_threshold=args.diff_threshold,
                               num_comparison=args.num_comparison,
                               num_worker=args.num_worker,
                               num_machine=args.num_machine,
                               id_machine=args.id_machine,
                               )
        if args.save_graph_file is not None:
            filename = args.save_graph_file
        else:
            filename = 'data/temp/temp_%d_%d.mg.graph.gzip'%(args.num_machine, args.id_machine)
        motion_graph.save_graph_only(filename)

    if args.analysis == "true":
        assert args.mode == 'load'

        np.random.seed(0)
        random.seed(0)
        motion_graph.clear_visit_info()
        motion_graph.verbose=False
        motion_length = 10.0
        visited_nodes = []
        visited_path = {}
        t_processed_total = 0.0

        num_motions_max = 8192*256
        checkpoints = [1024, 2048, 4096, 8192, 8192*2, 8192*4, 8192*8, 8192*16, 8192*32, 8192*64, 8192*128, 8192*256, 8192*512, 8192*1024, 8192*2048]

        graph = motion_graph.graph

        print('---------------------------------')
        print('checkpoint\ttime(hours)\tnodes\tedges')
        while len(visited_nodes) < num_motions_max:    
            
            while True:
                nodes, t_processed = motion_graph.create_random_path(
                    length=motion_length, leave_visit_info=True, use_visit_info="none")
                if t_processed >= motion_length:# and str(nodes) not in visited_path:
                    visited_nodes.append(nodes)
                    # visited_path[str(nodes)] = True
                    t_processed_total += t_processed
                    break

            if len(visited_nodes) in checkpoints:
                ''' Stat for nodes '''
                log_file = open('data/temp/mg_stat_nodes_%d.txt'%len(visited_nodes), 'w+')
                text = "node\tnum_visit\tnum_visit_norm\n"
                num_unvisited_nodes = 0
                for n in graph.nodes:
                    ''' Number of visit per node '''
                    num_visit = graph.nodes[n]['num_visit']
                    ''' Number of visit per node normalized by its in-degree '''
                    num_visit_normalized = num_visit / float(max(graph.in_degree(n), 1))
                    text += "%s\t%d\t%f\n"%(str(n), num_visit, num_visit_normalized)
                    if num_visit==0: num_unvisited_nodes += 1
                log_file.write(text)
                log_file.close()
                ''' Stat for edges '''
                log_file = open('data/temp/mg_stat_edges_%d.txt'%len(visited_nodes), 'w+')
                text = "edge\tnum_visit\n"
                num_unvisited_edges = 0
                for e in graph.edges:
                    ''' Number of visit per edge '''
                    num_visit = graph.edges[e]['num_visit']
                    text += "%s\t%d\n"%(str(e), num_visit)
                    if num_visit==0: num_unvisited_edges += 1
                log_file.write(text)
                log_file.close()

                percent_unvisited_nodes = 100*num_unvisited_nodes/float(graph.number_of_nodes())
                percent_unvisited_edges = 100*num_unvisited_edges/float(graph.number_of_edges())
                print('%d\t%f\t%f\t%f'%(len(visited_nodes), t_processed_total/3600.0, percent_unvisited_nodes, percent_unvisited_edges))
        print('---------------------------------')

        with gzip.open('data/temp/visited_nodes.mg.gzip', "wb") as f:
            pickle.dump(visited_nodes, f)
        exit(0)

    if args.create_motion == "true":
        assert args.mode == 'load'

        motion_graph.verbose=False
        mp.shared_data = motion_graph
        visited_nodes = mp.run_parallel_async_idx(create_random_motions, 
                                                  args.num_worker,
                                                  args.create_motion_num_motions,
                                                  args.create_motion_filename,
                                                  args.create_motion_max_length,
                                                  args.create_motion_min_length,
                                                  )

        with gzip.open('data/temp/visited_nodes.mg.gzip', "wb") as f:
            pickle.dump(visited_nodes, f)
        exit(0)

    # nx.draw(motion_graph.graph, node_color='green', with_labels=True)

    # cnt = 0
    # largest_wcc = max(nx.weakly_connected_components(motion_graph.graph), key=len)
    # largest_scc = max(nx.strongly_connected_components(motion_graph.graph), key=len)
    
    # print(motion_graph.graph.number_of_nodes(), len(largest_wcc), len(largest_scc))

    # nx.draw(motion_graph.graph.subgraph(largest_scc), node_color='red', with_labels=True)
    # plt.show()
    # for g in nx.strongly_connected_components(motion_graph.graph):
    #     print(g.number_of_nodes())
    #     # nx.draw(h, node_color='red', with_labels=True)
    # for g in nx.weakly_connected_components(motion_graph.graph):
    #     print(g.number_of_nodes())
    # exit(0)

    if args.visualize=="true":

        import matplotlib.pyplot as plt

        if args.graph_reduce is not None:
            motion_graph.reduce(args.graph_reduce)
        nn = motion_graph.graph.number_of_nodes()
        ne = motion_graph.graph.number_of_edges()
        print('The graph was reduced with %d nodes and %d edges'%(nn, ne))
        
        nx.draw(motion_graph.graph, with_labels=True)
        plt.show()

        motion, motion_raw = motion_graph._create_random_motion_debug(length=10.0)
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
        
        if np.allclose(char_info.v_up_env, np.array([0, 1, 0])):
            cam_pos = cam_origin + np.array([0.0, 2.0, -3.0])
            cam_vup = np.array([0.0, 1.0, 0.0])
        elif np.allclose(char_info.v_up_env, np.array([0, 0, 1])):
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

