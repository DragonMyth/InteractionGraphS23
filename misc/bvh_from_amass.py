'''
Descrition:
    This module extracts BVH files from AMASS data set.
    The average body shape will be used for the skeleton of BVH files.
Examples:
    python3 bvh_from_amass.py --files xxx
    python3 bvh_from_amass.py --dirs xxx --fps 30 --fix_height
'''


import sys, os
import numpy as np
import torch

from fairmotion.ops import conversions
from fairmotion.ops import math
from fairmotion.core import motion as mo_core
from fairmotion.ops import motion as mo_ops

from fairmotion.data import amass
from fairmotion.data import bvh

from fairmotion.utils import utils
# from fairmotion.utils import multiprocessing as mp
from fairmotion.viz.utils import TimeChecker

from heapq import nsmallest

import argparse

import re

# def auto_labels(job_idx, h_min_max=1.0, h_min_thres=0.2, duration=1.0):
#     res = []
#     num_jobs = job_idx[1] - job_idx[0]
#     if num_jobs <= 0: return res

#     labels = []

#     def get_heights(pose):
#         heights = []
#         for j in pose.skel.joints:
#             p = conversions.T2p(pose.get_transform(j, local=False))
#             p = math.projectionOnVector(p, pose.skel.v_up_env)
#             heights.append(np.linalg.norm(p))
#         return heights

#     for i in range(job_idx[0], job_idx[1]):
#         m = mp.shared_data[i]
#         t_remaining = duration
#         t_prev = m.times[0]
#         use = True
#         # print('=================================')
#         # print(files[i])
#         # print('=================================')
#         for frame in range(m.num_frame()):
#             pose = m.get_pose_by_frame(frame)
#             heights = get_heights(pose)
#             idx = np.argpartition(heights, 2)[:2]
#             h_min = np.mean(np.array(heights)[idx])
#             # print(h_min)
#             ''' 
#             If the minimum height is too high, there should be some stairs 
#             '''
#             if h_min >= h_min_max: 
#                 use = False
#                 break
#             ''' 
#             If the minimum height is above then h_min_thres for duration, 
#             there should be some stairs 
#             '''
#             if frame > 0:
#                 if h_min >= h_min_thres:
#                     t_remaining -= (m.times[frame] - t_prev)
#                 else:
#                     t_remaining = duration
#                 if t_remaining <= 0:
#                     use = False
#                     break
#             t_prev = m.times[frame]
#         labels.append(use)

#     return labels

def arg_parser():
    parser = argparse.ArgumentParser()
    ''' Files to process '''
    parser.add_argument(
        '--files', action='append', default=[])
    ''' Directories to process '''
    parser.add_argument(
        '--dirs', action='append', type=str, default=[])
    ''' FPS of output motions, the orignal motions will be resampled '''
    parser.add_argument(
        '--fps', type=int, default=30)
    ''' Fix height so that output motions do not penetrate the ground '''
    parser.add_argument(
        '--fix_height', action='store_true')
    ''' Minimum height for fixing motions '''
    parser.add_argument(
        '--min_height', type=float, default=0.05)
    ''' Ground plane up vector '''
    parser.add_argument(
        '--vup_env', type=str, default='z')
    ''' Fix height so that output motions do not penetrate the ground '''
    parser.add_argument(
        '--verbose', action='store_true')
    ''' Fix height so that output motions do not penetrate the ground '''
    parser.add_argument(
        '--bm_path', type=str, default='data/character/amass/smplh/male/model.npz')
    return parser

if __name__ == '__main__':

    args = arg_parser().parse_args()

    files = args.files
    for d in args.dirs:
        files += utils.files_in_dir(
            d, ext=".npz", keywords_exclude=["shape.npz"])
        files = np.random.permutation(files)

    assert len(files) > 0, 'No file to process'

    if args.verbose:
        print('+++++++++++++++++FilesToProcess+++++++++++++++++')
        print(files)
        print('++++++++++++++++++++++++++++++++++++++++++++++++')

    from human_body_prior.body_model.body_model import BodyModel
    bm = BodyModel(
        args.bm_path, 
        num_betas=10, 
        # model_type="smplh"
    ).to(torch.device("cpu"))

    for f in files:
        motion = amass.load(f, bm=bm, override_betas=np.zeros(10))
        mo_ops.resample(motion, fps=args.fps)
        if args.fix_height:
            '''
            Move the whole motion vertically so that 
            the feet do not penetrate the ground
            '''
            if args.vup_env=='y':
                h_idx = 1
            elif args.vup_env=='z':
                h_idx = 2
            else:
                raise NotImplementedError
            hs = []
            for i in range(motion.num_frames()):
                pose = motion.get_pose_by_frame(i)
                for j in range(pose.skel.num_joints()):
                    p = conversions.T2p(pose.get_transform(j, local=False))
                    hs.append(p[h_idx])
            
            if len(hs) > 5:
                ''' Pick 5 lowest frames then average them '''
                h_min = np.mean(nsmallest(5, hs))
            else:
                h_min = np.min(hs)

            h_pen = args.min_height - h_min
            h_offset = np.zeros(3)
            h_offset[h_idx] = h_pen
            mo_ops.translate(motion, h_offset)
        name, ext = os.path.splitext(f)
        bvh.save(motion, (name+".bvh"))
        if args.verbose:
            print(f, '->', (name+".bvh"))
