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
def arg_parser():
    parser = argparse.ArgumentParser()
    ''' List of files to read '''
    parser.add_argument('--files', action='append', default=[])
    parser.add_argument('--save_files', action='append', default=[])
    
    parser.add_argument('--fps',type=int,default=30)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--v_up_skel', type=str, default='y')
    parser.add_argument('--v_face_skel', type=str, default='z')
    parser.add_argument('--v_up_env', type=str, default='z')

    ''' List of directories to read '''
    parser.add_argument('--file_dir', action='append', type=str, default=None)
    parser.add_argument('--save_file_dir', action='append', type=str, default=None)

    parser.add_argument('--save_ext', action='append', type=str, default=None)


    parser.add_argument('--file_num', action='append', type=int, default=None)

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
            args.files += files
    print("Files: ",args.files)
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

    for i,motion in enumerate(motions):
        resampled_motion = motion_ops.resample(motion,args.fps)
        base_name = os.path.basename(args.files[i])
        base_name = os.path.splitext(base_name)[0]
        file_name = "%s/%s_%s.bvh"%(args.save_file_dir[0],base_name,args.save_ext[0])
        print('Saving Resampled: %s',file_name)

        bvh.save(motion, file_name)
        print('Save Done')

    # for i, motion in enumerate(motions):
    #     if i < len(translations):
    #         motion_ops.translate(
    #             motion, -motion.get_pose_by_time(0.0).get_facing_position())
    #         motion_ops.translate(
    #             motion, translations[i])
    #     if args.color is not None:
    #         character_colors.append(convert_args_to_array(args.color[i])/255.0)