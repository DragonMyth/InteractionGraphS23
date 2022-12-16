'''
python3 simple_bvh_viewer.py --files data/motion/amass/CMU/01/01_01_poses.bvh --body_model bullet_box
'''

import os
import sys
import numpy as np

from fairmotion.core.motion import Motion
from fairmotion.core.velocity import MotionWithVelocity
from fairmotion.ops import conversions, math, motion as motion_ops
from fairmotion.data import bvh
from fairmotion.utils import utils
from fairmotion.core import similarity

import argparse
import collections
import re
import random
import itertools
import pickle

from collections import deque

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(directory))

start_time = None

character_colors = [
    np.array([85,  160, 173, 255])/255,
    np.array([30,  120, 180, 255])/255,
    np.array([215, 40,  40,  255])/255,
    np.array([150, 100, 190, 255])/255,
    np.array([225, 120, 190, 255])/255,
    np.array([140, 90,  80,  255])/255,
    np.array([50,  160, 50,  255])/255,
    np.array([255, 125, 15,  255])/255,
    np.array([125, 125, 125, 255])/255,
    np.array([255, 0,   255, 255])/255,
    np.array([0,   255, 125, 255])/255,
    np.array([50,  50,  50,  255])/255,
    np.array([175, 175, 175, 255])/255,
    ]

character_colors = [
    np.array([30,  120, 180, 255])/255,
    np.array([215, 40,  40,  255])/255,
    np.array([150, 100, 190, 255])/255,
    np.array([225, 120, 190, 255])/255,
    np.array([140, 90,  80,  255])/255,
    np.array([50,  160, 50,  255])/255,
    np.array([255, 125, 15,  255])/255,
    np.array([125, 125, 125, 255])/255,
    np.array([255, 0,   255, 255])/255,
    np.array([0,   255, 125, 255])/255,
    np.array([50,  50,  50,  255])/255,
    np.array([175, 175, 175, 255])/255,
    np.array([248, 215, 3,   255])/255,
    np.array([248, 60,  18,  255])/255,
    np.array([243, 118, 97,  255])/255,
    np.array([247, 116, 25,  255])/255,
    np.array([249, 241, 215, 255])/255,
    ]

# character_colors = None

# character_colors = [
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
#     np.array([85,  160, 173, 255])/255,
# ]

character_color_end_of_motion = np.array([255,  80, 80, 255])/255

SCREENSHOT_DIR = 'data/screenshot/'

def arg_parser():
    parser = argparse.ArgumentParser()
    ''' List of files to read '''
    parser.add_argument('--files', action='append', default=[])
    ''' List of directories to read '''
    parser.add_argument('--file_dir', action='append', type=str, default=None)
    parser.add_argument('--file_num', action='append', type=int, default=None)
    parser.add_argument("--file_shuffle", action='store_true')
    parser.add_argument('--cluster_info', type=str, default=None)
    parser.add_argument('--cluster_id', action='append', type=int, default=None)
    parser.add_argument('--cluster_num_sample', action='append', type=int, default=None)
    parser.add_argument('--cluster_sample_method', type=str, default="random")
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--v_up_skel', type=str, default='y')
    parser.add_argument('--v_face_skel', type=str, default='z')
    parser.add_argument('--v_up_env', type=str, default='z')
    parser.add_argument('--body_model', choices=["line", "cylinder", "bullet"], default='cylinder')
    parser.add_argument('--bullet_char_file', type=str, default='data/character/amass.urdf')
    parser.add_argument('--bullet_char_info', type=str, default='amass_char_info.py')
    parser.add_argument('--translation', nargs='+', action='append', type=str, default=None)
    parser.add_argument("--grid", action='store_true')
    parser.add_argument('--grid_width', type=float, default=1.0)
    parser.add_argument('--grid_height', type=float, default=1.0)
    parser.add_argument('--color', nargs='+', action='append', type=int, default=None)
    parser.add_argument('--random_start', type=float, default=None)
    parser.add_argument('--plane_texture', type=str, default="data/image/grid2.png")
    parser.add_argument('--render_window_w', type=int, default=1280)
    parser.add_argument('--render_window_h', type=int, default=720)
    parser.add_argument('--render_overlay', type=str, default='true')
    parser.add_argument('--cam_pos', nargs='+', type=float, default=None)
    parser.add_argument('--cam_origin', nargs='+', type=float, default=None)
    parser.add_argument('--cam_fov', type=float, default=45.0)
    parser.add_argument('--cam_follow', type=str, default='true')
    parser.add_argument('--cam_file', type=str, default=None)
    parser.add_argument('--screenshot_dir', type=str, default=None)
    parser.add_argument('--screenshot_start_time', type=float, default=None)
    parser.add_argument('--screenshot_end_time', type=float, default=None)
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument('--ground_w', type=float, default=100.0)
    parser.add_argument('--ground_h', type=float, default=100.0)
    parser.add_argument('--ground_tile_w', type=float, default=2.0)
    parser.add_argument('--ground_tile_h', type=float, default=2.0)
    parser.add_argument("--frame_diff_print", action='store_true')
    return parser

def convert_args_to_array(arg):
    return np.array([float(arg[0]),float(arg[1]),float(arg[2])])

def str2bool(string):
    if string=="true": 
        return True
    elif string=="false": 
        return False
    else:
        raise Exception('Unknown')

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        words = None
        with open(sys.argv[1], 'rb') as f:
            words = [word.decode() for line in f for word in line.split()]
        if words is None:
            raise Exception('Invalid Argument')
        args = arg_parser().parse_args(words)
    else:
        args = arg_parser().parse_args()
    
    ''' Load Motions '''

    motions = []
    file_idx = 0

    if args.file_dir is not None:
        files = []
        for i, d in enumerate(args.file_dir):
            if args.file_num is not None:
                files += utils.files_in_dir(d, 
                                            ext=".bvh", 
                                            sort=True, 
                                            sample_mode="sequential", 
                                            sample_num=args.file_num[i])
            else:
                files += utils.files_in_dir(d, 
                                            ext=".bvh", 
                                            sort=True)
        if args.cluster_info is not None:
            cluster_info = []
            files_filtered = []
            files_cluster_id = []
            with open(args.cluster_info, 'r') as file:
                for line in file:
                    l = re.split('[\t|\n|,|:| ]+', line)
                    cluster_id, rank, score, filename = int(l[0]), int(l[1]), float(l[2]), str(l[3])
                    cluster_info.append((cluster_id, rank, score, filename))

            files_in_clusters = {}
            for cluster_id, rank, score, filename in cluster_info:
                if cluster_id not in files_in_clusters: files_in_clusters[cluster_id] = []
                files_in_clusters[cluster_id].append('%s/%s'%(args.file_dir,filename))
            num_cluster = len(files_in_clusters.keys())

            print('-------Motions are sampled from-------')
            num_cluster_used = 0
            cluster_id_used = []
            assert args.cluster_id is not None
            assert args.cluster_num_sample is not None
            assert len(args.cluster_id) == len(args.cluster_num_sample)
            for i, cluster_id in enumerate(args.cluster_id):
                files = files_in_clusters[cluster_id]
                num_sample = args.cluster_num_sample[i]
                if num_sample == 0:
                    continue
                elif num_sample < 0:
                    files_filtered += files
                    files_cluster_id += [cluster_id]*len(files)
                else:
                    if args.cluster_sample_method == "random":
                        files_filtered += random.choices(files, k=num_sample)
                    elif args.cluster_sample_method == "top":
                        files_filtered += files[:num_sample]
                    files_cluster_id += [cluster_id]*num_sample
                num_cluster_used += 1
                cluster_id_used.append(cluster_id)
                print("cluster[%d], (%d / %d)"%(cluster_id, num_sample, len(files)))
            files = files_filtered
        
        args.files += files

    if args.file_shuffle:
        random.shuffle(args.files)

    translations = []
    
    if args.color is not None:
        character_colors = []

    if args.random_start is not None:
        assert args.random_start >= 0.0
        start_time = np.random.uniform(0, args.random_start, size=len(args.files))

    if args.translation is not None:
        assert not args.grid
        assert len(args.translation) >= len(args.files)
        translations = [convert_args_to_array(args.translation[i]) for i in range(len(args.files))]

    if args.grid:
        assert args.translation is None
        area_per_character = (args.grid_width * args.grid_height) / len(args.files)
        grid_stride = np.sqrt(area_per_character)
        num_w = int(args.grid_width/grid_stride)
        num_h = int(args.grid_height/grid_stride)
        while num_w * num_h < len(args.files):
            if num_h < num_w: 
                num_h += 1
            else:
                num_w += 1
        xs = list(np.linspace(-0.5*args.grid_width, 0.5*args.grid_width, num_w, endpoint=True))
        ys = list(np.linspace(-0.5*args.grid_height, 0.5*args.grid_height, num_h, endpoint=True))
        xys = list(itertools.product(*[xs, ys]))
        if args.v_up_env == 'y':
            translations = [np.array([x, 0, y]) for x, y in xys]
        else:
            translations = [np.array([y, x, 0]) for x, y in xys]

    if args.num_worker > 1:
        motions = kinematics.read_motion_parallel(
            args.files,
            args.num_worker, 
            args.scale, 
            kinematics.str_to_axis(args.v_up_skel), 
            kinematics.str_to_axis(args.v_face_skel),
            kinematics.str_to_axis(args.v_up_env),
            True,
            30.0,
            True)
    else:
        for file in args.files:
            motion = bvh.load(
                file=file,
                scale=args.scale,
                v_up_skel=utils.str_to_axis(args.v_up_skel), 
                v_face_skel=utils.str_to_axis(args.v_face_skel), 
                v_up_env=utils.str_to_axis(args.v_up_env))
            if args.frame_diff_print:
                motion = MotionWithVelocity.from_motion(motion)
            motions.append(motion)
            print('Loaded:', file)

    for i, motion in enumerate(motions):
        if i < len(translations):
            motion_ops.translate(
                motion, -motion.get_pose_by_time(0.0).get_facing_position())
            motion_ops.translate(
                motion, translations[i])
        if args.color is not None:
            character_colors.append(convert_args_to_array(args.color[i])/255.0)

    assert len(motions) > 0

    from fairmotion.viz.utils import TimeChecker
    from fairmotion.viz import glut_viewer
    from fairmotion.viz import camera
    import render_module as rm
    rm.initialize()

    cam = None
    if args.cam_file is not None:
        with gzip.open(args.cam_file, "rb") as f:
            cam = pickle.load(f)
    else:
        if args.cam_origin is not None:
            flag['follow_cam'] = False
            cam_origin = convert_args_to_array(args.cam_origin)
        else:
            cam_origin = conversions.T2p(motion.get_pose_by_time(0.0).get_root_transform())

        if args.v_up_env == 'y':
            cam_vup = np.array([0.0, 1.0, 0.0])
            if args.cam_pos is not None:
                cam_pos = convert_args_to_array(args.cam_pos)
            else:
                cam_pos = cam_origin + np.array([0.0, 2.0, -3.0])
        elif args.v_up_env == 'z':
            cam_vup = np.array([0.0, 0.0, 1.0])
            if args.cam_pos is not None:
                cam_pos = convert_args_to_array(args.cam_pos)
            else:
                cam_pos = cam_origin + np.array([-3.0, 0.0, 2.0])
        else:
            raise NotImplementedError

        cam = camera.Camera(pos=cam_pos,
                            origin=cam_origin, 
                            vup=cam_vup, 
                            fov=args.cam_fov)

    class MotionViewer(glut_viewer.Viewer):
        def __init__(
            self, 
            args,
            motions,
            start_time,
            character_colors=None,
            title="simple_bvh_viewer", 
            cam=None, 
            size=(1280, 720)):
            
            super().__init__(title, cam, size)
            self.motions = motions
            self.args = args
            self.character_colors = character_colors
            self.rm = rm
            self.tex_id_ground = None
            assert self.args.body_model in ["line", "cylinder", "bullet"]
            if self.args.body_model == "bullet":
                import importlib.util
                import pybullet as pb
                import pybullet_data
                import sim_agent
                from bullet import bullet_client
                from bullet import bullet_render
                self.pb_client = bullet_client.BulletClient(
                    connection_mode=pb.DIRECT, options=' --opengl2')
                self.pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
                spec = importlib.util.spec_from_file_location("bullet_char_info", self.args.bullet_char_info)
                bullet_char_info = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bullet_char_info)
                self.bullet_agent = sim_agent.SimAgent(
                    pybullet_client=self.pb_client, 
                    model_file=self.args.bullet_char_file, 
                    char_info=bullet_char_info,
                    ref_scale=self.args.scale,
                    self_collision=False,
                    kinematic_only=True,
                    verbose=False)

            self.ground_size = (self.args.ground_w, self.args.ground_h)
            self.ground_tile_size = (self.args.ground_tile_w, self.args.ground_tile_h)

            assert args.v_up_env in ['y', 'z']
            self.v_up_env = np.array([0, 1, 0]) if args.v_up_env == 'y' else np.array([0, 0, 1])

            self.file_idx = 0
            self.cur_time = 0.0
            self.start_time = start_time

            # Flags
            self.flag = {}
            self.flag['follow_cam'] = True
            self.flag['ground'] = True
            self.flag['origin'] = False
            self.flag['joint_xform'] = False
            self.flag['facing_xform'] = False
            self.flag['root_trajectory'] = False
            self.flag['auto_play'] = False
            self.flag['render_all_motions'] = False
            self.flag['fog'] = False
            self.flag['overlay'] = True
            self.flag['history'] = False

            self.toggle = {}
            self.toggle[b'0'] = 'follow_cam'
            self.toggle[b'1'] = 'ground'
            self.toggle[b'2'] = 'origin'
            self.toggle[b'3'] = 'joint_xform'
            self.toggle[b'4'] = 'facing_xform'
            self.toggle[b'5'] = 'root_trajectory'
            self.toggle[b'6'] = 'render_all_motions'
            self.toggle[b'0'] = 'overlay'
            self.toggle[b'a'] = 'auto_play'
            self.toggle[b'f'] = 'fog'
            self.toggle[b'h'] = 'history'

            self.time_checker_auto_play = TimeChecker()
            self.time_checker_global = TimeChecker()

            self.play_speed = 1.0

            self.render_return_times = deque(maxlen=10)

            print('===== Simple BVH Viewer =====')

        def render_callback(self):
            self.render_return_times.append(self.time_checker_global.get_time())
            
            if self.flag['ground']:
                self.rm.gl.glDisable(self.rm.gl.GL_BLEND)
                if self.tex_id_ground is None:
                    self.tex_id_ground = \
                        self.rm.gl_render.load_texture(self.args.plane_texture)
                self.rm.gl_render.render_ground_texture(
                    self.tex_id_ground,
                    size=self.ground_size, 
                    dsize=self.ground_tile_size, 
                    axis=args.v_up_env,
                    origin=self.flag['origin'],
                    use_arrow=True)

            if self.flag['render_all_motions']:
                motions_to_draw = self.motions
            else:
                motions_to_draw = [self.motions[self.file_idx]]

            self.render_characters(
                motions_to_draw, 
                self.cur_time, 
                self.start_time, 
                self.character_colors)

            if self.flag['fog']:
                density = 0.05;
                fogColor = [1.0, 1.0, 1.0, 0.2]
                self.rm.gl.glEnable(self.rm.gl.GL_FOG)
                self.rm.gl.glFogi(self.rm.gl.GL_FOG_MODE, self.rm.gl.GL_EXP2)
                self.rm.gl.glFogfv(self.rm.gl.GL_FOG_COLOR, fogColor)
                self.rm.gl.glFogf(self.rm.gl.GL_FOG_DENSITY, density)
                self.rm.gl.glHint(self.rm.gl.GL_FOG_HINT, self.rm.gl.GL_NICEST)
            else:
                self.rm.gl.glDisable(self.rm.gl.GL_FOG)
        
        def overlay_callback(self):            
            if not self.flag['overlay']: return

            motion = self.motions[self.file_idx]

            self.rm.gl.glPushAttrib(self.rm.gl.GL_LIGHTING)
            self.rm.gl.glDisable(self.rm.gl.GL_LIGHTING)
            
            w, h = self.window_size
            font = self.rm.glut.GLUT_BITMAP_9_BY_15
            ''' File name '''
            self.rm.gl_render.render_text(
                "File: %s"%self.args.files[self.file_idx], pos=[0.05*w, 0.05*h], font=font)
            ''' FPS '''
            h_start = h-100
            self.rm.gl_render.render_text(
                "FPS: %.2f"%self.get_avg_fps(), pos=[0.05*w, h_start], font=font)
            self.rm.gl_render.render_text(
                "Time: %.2f / %.2f"%(self.cur_time, motion.length()), pos=[0.05*w, h_start+20], font=font)
            self.rm.gl_render.render_text(
                "Frame: %d"%(motion.time_to_frame(self.cur_time)), pos=[0.05*w, h_start+40], font=font)
            self.rm.gl_render.render_text(
                "Playspeed: x %.2f"%(self.play_speed), pos=[0.05*w, h_start+60], font=font)
            self.rm.gl_render.render_text(
                "Size: %d x %d"%(w,h), pos=[0.05*w, h_start+80], font=self.rm.glut.GLUT_BITMAP_9_BY_15)
            self.rm.gl.glPopAttrib(self.rm.gl.GL_LIGHTING)

        def keyboard_callback(self, key):
            global args
            global link_info_line_width

            flag = self.flag
            toggle = self.toggle
            motion = self.motions[file_idx]

            if key in toggle:
                flag[toggle[key]] = not flag[toggle[key]]
                print('Toggle:', toggle[key], flag[toggle[key]])
                link_info_line_width = 2.0 if not flag['render_all_motions'] else 0.01
            elif key == b'h':
                self.rm.glut.glutHideWindow()
            elif key == b'H':
                self.rm.glut.glutShowWindow()
            elif key == b'r':
                self.cur_time = 0.0
                self.time_checker_auto_play.begin()
                self.time_checker_global.begin()
            elif key == b']':
                next_frame = min(
                    motion.num_frames()-1,
                    motion.time_to_frame(self.cur_time+motion.fps_inv))
                self.cur_time = motion.frame_to_time(next_frame)
                if args.frame_diff_print:
                    prev_frame = max(0, next_frame-1)
                    pose_i = motion.get_pose_by_frame(prev_frame)
                    pose_j = motion.get_pose_by_frame(next_frame)
                    vel_i = motion.get_velocity_by_frame(prev_frame)
                    vel_j = motion.get_velocity_by_frame(next_frame)
                    T_ref_i = pose_i.get_facing_transform()
                    T_ref_j = pose_j.get_facing_transform()
                    w_joint_pos = 1.0
                    w_joint_vel = 1.0
                    w_joints = None
                    w_root_pos = 1.0
                    w_root_vel = 1.0
                    w_ee_pos = 1.0
                    w_ee_vel = 1.0
                    print("------------------------------")
                    similarity.pose_similarity(
                        pose_i,
                        pose_j,
                        vel_i,
                        vel_j,
                        w_joint_pos, 
                        w_joint_vel, 
                        w_joints,
                        verbose=True,
                    )
                    similarity.root_ee_similarity(
                        pose_i,
                        pose_j,
                        vel_i,
                        vel_j,
                        w_root_pos,
                        w_root_vel,
                        w_ee_pos,
                        w_ee_vel,
                        T_ref_i,
                        T_ref_j,
                        verbose=True,
                    )
            elif key == b'[':
                prev_frame = max(0,
                                 motion.time_to_frame(self.cur_time-motion.fps_inv))
                self.cur_time = motion.frame_to_time(prev_frame)
            elif key == b'+':
                self.play_speed = min(self.play_speed+0.2, 5.0)
                # print('play_speed:', play_speed)
            elif key == b'-':
                self.play_speed = max(self.play_speed-0.2, 0.2)
                # print('play_speed:', play_speed)
            elif key == b'h':
                pose = motion.get_pose_by_frame(motion.time_to_frame(self.cur_time))
                heights = []
                for j in pose.skel.joints:
                    p = conversions.T2p(pose.get_transform(j, local=False))
                    p = math.projectionOnVector(p, pose.skel.v_up_env)
                    heights.append(np.linalg.norm(p))
                print(np.min(heights), np.max(heights))
            elif key == b',':
                file_idx_prev = self.file_idx
                self.file_idx = max(0, self.file_idx-1)
                if file_idx_prev != self.file_idx: self.cur_time = 0.0
            elif key == b'.':
                file_idx_prev = self.file_idx
                self.file_idx = min(len(motions)-1, self.file_idx+1)
                if file_idx_prev != self.file_idx: self.cur_time = 0.0
            elif key == b'M':
                filename = 'data/temp/temp.cam'
                with open(filename, "wb") as file:
                    pickle.dump(self.cam_cur, file)
                    print('Saved:', filename)
            elif key == b'm':
                if args.cam_file is not None:
                    filename = args.cam_file
                else:
                    filename = 'data/temp/temp.cam'
                with open(filename, "rb") as file:
                    self.cam_cur = pickle.load(file)
                    print('Loaded:', filename)
            elif key == b'/':
                flag['follow_cam'] = False
                cam_origin = np.zeros(3)
                # if args.v_up_env=='y':
                #     cam_pos = cam_origin + np.array([0.0, 18.0, -16.0])
                # elif args.v_up_env=='z':
                #     cam_pos = cam_origin + np.array([16.0, 0.0, 18.0])
                # self.cam_cur.pos = cam_pos
                # self.cam_cur.origin = cam_origin
                self.cam_cur.pos = np.array([10.24743462, 26.76471575, -1.55535087])
                self.cam_cur.origin = np.array([-2.15004782,  1.04735372, -1.54255776])
            elif key == b'?':
                print('Cam pos:', self.cam_cur.pos)
                print('Cam origin:', self.cam_cur.origin)
            elif key == b'c' or key == b'C':
                if key==b'C':
                    if flag['render_all_motions']:
                        print("Please turn off the flag: render_all_motions")
                        return
                if args.screenshot_start_time is None:
                    start_time = input("Enter start time (sec): ")
                    try:
                       start_time = float(start_time)
                    except ValueError:
                       print("That's not a number!")
                       return
                else:
                    start_time = args.screenshot_start_time

                if args.screenshot_end_time is None:
                    end_time = input("Enter end time (sec): ")
                    try:
                       end_time = float(end_time)
                    except ValueError:
                       print("That's not a number!")
                       return
                else:
                    end_time = args.screenshot_end_time
                
                if args.screenshot_dir is None:
                    suffix = input("Enter subdirectory for screenshot file: ")
                    save_dir = "%s/%s"%(SCREENSHOT_DIR, suffix)
                else:
                    save_dir = args.screenshot_dir
                
                try:
                    os.makedirs(save_dir, exist_ok = True)
                except OSError:
                    print("Invalid Subdirectory")
                    return

                if args.cam_file is not None:
                    self.load_cam(args.cam_file)

                file_idx_prev = self.file_idx
                self.file_idx = min(len(motions)-1, self.file_idx+1)
                if file_idx_prev != self.file_idx: self.cur_time = 0.0

                if key==b'C':
                    motion_indicies_to_draw = [i for i in range(len(motions))]
                else:
                    motion_indicies_to_draw = [self.file_idx]

                for idx in motion_indicies_to_draw:
                    if len(motion_indicies_to_draw) > 1:
                        save_dir_i = os.path.join(save_dir, '%02d'%idx)
                        os.mkdir(save_dir_i)
                    else:
                        save_dir_i = save_dir
                    cnt_screenshot = 0
                    time_processed = start_time
                    self.cur_time = start_time
                    self.file_idx = idx
                    motion = self.motions[self.file_idx]
                    dt = 1/30.0
                    print("Processing:", save_dir_i)
                    while self.cur_time <= end_time and self.cur_time < motion.length():
                        name = 'screenshot_%04d'%(cnt_screenshot)
                        p = conversions.T2p(
                            motion.get_pose_by_time(self.cur_time).get_root_transform())
                        if not flag['render_all_motions'] and flag['follow_cam']:
                            if args.v_up_env == 'y':
                                self.update_target_pos(p, ignore_y=True)
                            elif args.v_up_env == 'z':
                                self.update_target_pos(p, ignore_z=True)
                            else:
                                raise NotImplementedError
                        self.save_screen(dir=save_dir_i, name=name, render=True)
                        print('\rtime_elased:', self.cur_time, '(', name, ')', end=' ')
                        self.cur_time += dt
                        cnt_screenshot += 1
                    os.system(f"ffmpeg -r 30 -i {save_dir_i}/screenshot_%04d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {save_dir_i}_render.mp4")
                    print("\n")
            elif key == b's':
                try:
                    save_dir = input("Enter subdirectory for motions: ")
                    os.makedirs(save_dir, exist_ok = True)
                    fps = int(input("Enter FPS: "))
                    scale = float(input("Enter scale: "))
                except OSError:
                    print("Invalid Directory")
                    return
                except ValueError:
                    print("That's not a number!")
                    return
                for i, file in enumerate(args.files):
                    head, tail = os.path.split(file)
                    bvh.save(
                        motion, 
                        os.path.join(save_dir, tail),
                        scale=scale, rot_order="XYZ", verbose=False)
            elif key == b'd':
                for i in range(len(motions)):
                    render_data = self.record_render_data(agent=self.bullet_agent, motion=motions[i])
                    with open(os.path.join(f"data/temp/render_data_{i}.pkl"), "wb") as file:
                        pickle.dump(render_data, file)
            else:
                return False
            return True

        def idle_callback(self):
            flag = self.flag
            motion = self.motions[self.file_idx]

            time_elapsed = self.time_checker_auto_play.get_time(restart=False)
            if flag['auto_play']:
                self.cur_time += self.play_speed * time_elapsed
            self.time_checker_auto_play.begin()

            if flag['follow_cam'] and not flag['render_all_motions']:
                time = min(self.cur_time, motion.length())
                p = conversions.T2p(motion.get_pose_by_time(time).get_root_transform())
                if np.allclose(motion.skel.v_up_env, np.array([0.0, 1.0, 0.0])):
                    self.update_target_pos(p, ignore_y=True)
                elif np.allclose(motion.skel.v_up_env, np.array([0.0, 0.0, 1.0])):
                    self.update_target_pos(p, ignore_z=True)
                else:
                    raise NotImplementedError

        def render_pose(self, pose, body_model, color, flag):
            skel = pose.skel
            if body_model=='line':
                for j in skel.joints:
                    T = pose.get_transform(j, local=False)
                    pos = conversions.T2p(T)
                    self.rm.gl_render.render_point(
                        pos, radius=0.03, color=[0.8, 0.8, 0.0, 1.0])
                    if flag['joint_xform']:
                        self.rm.gl_render.render_transform(T, scale=0.1)
                    if j.parent_joint is not None:
                        pos_parent = conversions.T2p(
                            pose.get_transform(j.parent_joint, local=False))
                        self.rm.gl_render.render_line(p1=pos_parent, p2=pos, color=color)
            elif body_model=='cylinder':
                for j in skel.joints:
                    T = pose.get_transform(j, local=False)
                    pos = conversions.T2p(T)
                    self.rm.gl_render.render_point(pos, radius=0.03, color=color)
                    if flag['joint_xform']:
                        self.rm.gl_render.render_transform(T, scale=0.1)
                    if j.parent_joint is not None:
                        # returns X that X dot vec1 = vec2 
                        pos_parent = conversions.T2p(
                            pose.get_transform(j.parent_joint, local=False))
                        p = 0.5 * (pos_parent + pos)
                        l = np.linalg.norm(pos_parent-pos)
                        r = 0.05
                        R = math.R_from_vectors(np.array([0, 0, 1]), pos_parent-pos)
                        self.rm.gl_render.render_capsule(
                            conversions.Rp2T(R,p), l, r, color=color, slice=16)
                        # self.rm.gl_render.render_line(p1=pos_parent, p2=pos, color=color)

        def get_render_data(self, agent):
            pb_client = self.pb_client
            model = agent._body_id
            joint_data = []
            link_data = []
            for j in range(pb_client.getNumJoints(model)):
                joint_info = pb_client.getJointInfo(model, j)
                joint_local_p, joint_local_Q, link_idx = joint_info[14], joint_info[15], joint_info[16]
                T_joint_local = conversions.Qp2T(
                    np.array(joint_local_Q), np.array(joint_local_p))
                if link_idx == -1:
                    link_world_p, link_world_Q = pb_client.getBasePositionAndOrientation(model)
                else:
                    link_info = pb_client.getLinkState(model, link_idx)
                    link_world_p, link_world_Q = link_info[0], link_info[1]
                T_link_world = conversions.Qp2T(
                    np.array(link_world_Q), np.array(link_world_p))
                T_joint_world = np.dot(T_link_world, T_joint_local)
                R, p = conversions.T2Rp(T_joint_world)
                joint_data.append((conversions.R2Q(R), p))

            data_visual = pb_client.getVisualShapeData(model)
            lids = [d[1] for d in data_visual]
            dvs = data_visual
            for lid, dv in zip(lids, dvs):
                if lid == -1:
                    p, Q = pb_client.getBasePositionAndOrientation(model)
                else:
                    link_state = pb_client.getLinkState(model, lid)
                    p, Q = link_state[4], link_state[5]

                p, Q = np.array(p), np.array(Q)
                R = conversions.Q2R(Q)
                T_joint = conversions.Rp2T(R, p)
                T_visual_from_joint = \
                    conversions.Qp2T(np.array(dv[6]),np.array(dv[5]))
                R, p = conversions.T2Rp(np.dot(T_joint, T_visual_from_joint))
                link_data.append((conversions.R2Q(R), p))
            return joint_data, link_data

        def record_render_data(self, agent, motion):
            data = collections.defaultdict(list)
            for i, pose in enumerate(motion.poses):
                agent.set_pose(
                    pose,
                )
                joint_data, link_data = self.get_render_data(
                    agent=agent,
                )
                data['joint_data'].append(joint_data)
                data['link_data'].append(link_data)
            return data

        def render_characters(
            self, 
            motions, 
            cur_time, 
            start_time, 
            colors,
            character_color_end_of_motion=character_color_end_of_motion,
            history_render_interval=3.0,
            history_render_alpha=0.5,
            repetition=False):

            def render_pose(pose, color, link_info=True, shadow=True):
                if self.args.body_model=='line':
                    self.render_pose(pose, self.args.body_model, color, self.flag)
                elif self.args.body_model=='cylinder':
                    self.rm.gl.glEnable(self.rm.gl.GL_DEPTH_TEST)
                    if True:
                        self.rm.gl.glDisable(self.rm.gl.GL_LIGHTING)
                        self.rm.gl.glPushMatrix()
                        d = np.array([1, 1, 1])
                        d = d - math.projectionOnVector(d, self.v_up_env)
                        offset = 0.001 * self.v_up_env
                        self.rm.gl.glTranslatef(offset[0], offset[1], offset[2])
                        self.rm.gl.glScalef(d[0], d[1], d[2])
                        self.render_pose(pose, self.args.body_model, [0.5,0.5,0.5,1.0], self.flag)
                        self.rm.gl.glPopMatrix()
                    self.rm.gl.glEnable(self.rm.gl.GL_LIGHTING)
                    self.render_pose(pose, self.args.body_model, color, self.flag)
                elif self.args.body_model=='bullet':
                    self.rm.gl.glEnable(self.rm.gl.GL_DEPTH_TEST)
                    self.bullet_agent.set_pose(pose)
                    if shadow:
                        self.rm.gl.glPushMatrix()
                        d = np.array([1, 1, 1])
                        d = d - math.projectionOnVector(d, self.v_up_env)
                        offset = 0.001*self.v_up_env
                        self.rm.gl.glTranslatef(offset[0], offset[1], offset[2])
                        self.rm.gl.glScalef(d[0], d[1], d[2])
                        self.rm.bullet_render.render_model(
                            self.pb_client, 
                            self.bullet_agent._body_id, 
                            draw_link=True, 
                            draw_link_info=False, 
                            draw_joint=False, 
                            draw_joint_geom=False, 
                            ee_indices=None, 
                            color=[0.5,0.5,0.5,1.0],
                            lighting=False)
                        self.rm.gl.glPopMatrix()
                    self.rm.bullet_render.render_model(
                        self.pb_client, 
                        self.bullet_agent._body_id, 
                        draw_link=True, 
                        draw_link_info=link_info, 
                        draw_joint=False, 
                        draw_joint_geom=link_info, 
                        ee_indices=self.bullet_agent._char_info.end_effector_indices, 
                        color=color,
                        link_info_scale=1.01,
                        link_info_color=[0,0,0,1.0],
                        link_info_line_width=1.0,
                        lighting=True)
                else:
                    raise NotImplementedError

            for i, motion in enumerate(motions):
                time_offset = 0.0 if start_time is None else start_time[i]
                    
                if repetition:
                    t = (cur_time+time_offset) % motion.length()
                else:
                    t = min(cur_time+time_offset, motion.length())
                cur_pose = motion.get_pose_by_time(t)
                
                if colors is None:
                    # color = np.array([85, 160, 173, 255])/255.0
                    color = np.array([85, 160, 173, 255])/255.0
                else:
                    color = colors[min(i, len(colors)-1)]

                if character_color_end_of_motion is not None:
                    if cur_time+time_offset >= motion.length():
                        color = character_color_end_of_motion

                render_pose(cur_pose, color)

                if self.flag['facing_xform']:
                    T = pose.get_facing_transform()
                    self.rm.gl_render.render_transform(T, scale=0.5)

                if self.flag['root_trajectory']:
                    def render_trajectory(
                        points, color, scale=1.0, line_width=1.0, point_size=1.0):
                        self.rm.gl.glDisable(self.rm.gl.GL_LIGHTING)
                        self.rm.gl.glColor(color)
                        self.rm.gl.glLineWidth(line_width)
                        self.rm.gl.glBegin(self.rm.gl.GL_LINE_STRIP)
                        for p in points:
                            self.rm.gl.glVertex3d(p[0], p[1], p[2])
                        self.rm.gl.glEnd()
                    points = []
                    t_max = t
                    t_cur = time_offset
                    while t_cur <= t_max:
                        pose = motion.get_pose_by_frame(motion.time_to_frame(t_cur))
                        points.append(conversions.T2p(pose.get_root_transform()))
                        t_cur += 0.2
                    if len(points) > 1:
                        render_trajectory(
                            points, color, scale=1.0, line_width=5.0, point_size=1.0)

                if self.flag['history'] and not repetition:
                    self.rm.gl.glEnable(self.rm.gl.GL_BLEND)
                    poses = []
                    t = cur_time
                    alpha = 0.6
                    while True:
                        if t <= 0.0:
                            break
                        pose_hist = motion.get_pose_by_time(t)
                        color_hist = color.copy()
                        color_hist[3] = alpha
                        render_pose(pose_hist, color_hist, False, False)
                        t = max(0, t-history_render_interval)
                        alpha *= 0.95
                

        def get_avg_fps(self):
            if len(self.render_return_times) > 0:
                return int(1.0/np.mean(self.render_return_times))
            else:
                return 0
        def update_target_pos(self, pos, ignore_x=False, ignore_y=False, ignore_z=False):
            if np.array_equal(pos, self.cam_cur.origin):
                return
            d = pos - self.cam_cur.origin
            if ignore_x: d[0] = 0.0
            if ignore_y: d[1] = 0.0
            if ignore_z: d[2] = 0.0
            self.cam_cur.translate(d)
    renderer = MotionViewer(
        args=args,
        motions=motions,
        start_time=start_time,
        character_colors=character_colors,
        cam=cam,
        size=(args.render_window_w, args.render_window_h))
    renderer.run()

