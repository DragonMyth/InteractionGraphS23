'''
python3 creat_motion_from_AMASS.py --dir xxx
'''


import sys, os
import torch
import numpy as np

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from basecode.utils import multiprocessing as mp
from basecode.utils import basics
from basecode.motion import kinematics_simple as kinematics
from basecode.math import mmMath
from heapq import nsmallest

import argparse

import pickle
import gzip
import copy

import re

joint_names = [
    "root",
    "lhip",
    "rhip",
    "lowerback",
    "lknee",
    "rknee",
    "upperback",
    "lankle",
    "rankle",
    "chest",
    "ltoe",
    "rtoe",
    "lowerneck",
    "lclavicle",
    "rclavicle",
    "upperneck",
    "lshoulder",
    "rshoulder",
    "lelbow",
    "relbow",
    "lwrist",
    "rwrist",
]

# def convert_amass(m_amass):


# Choose the device to run the body model on.
comp_device = torch.device("cpu")

# Downloaded the required body model and put that in body_models directory of this repository.
# For SMPL-H download it from http://mano.is.tue.mpg.de/ and DMPLs you can obtain from http://smpl.is.tue.mpg.de/downloads.

from human_body_prior.body_model.body_model import BodyModel

bm_path = 'data/character/amass/smplh/male/model.npz'
# dmpl_path = '../body_models/dmpls/male/model.npz'

num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

# bm = BodyModel(bm_path=bm_path, num_betas=num_betas, num_dmpls=num_dmpls, path_dmpl=dmpl_path).to(comp_device)
bm = BodyModel(bm_path=bm_path, num_betas=num_betas).to(comp_device)
faces = c2c(bm.f)

npz_bdata_path = 'data/motion/amass/CMU/01/01_01_poses.npz' # the path to body data
bdata = np.load(npz_bdata_path)
print('Data keys available:%s'%list(bdata.keys()))
print('Vector poses has %d elements for each of %d frames.'%(bdata['poses'].shape[1], bdata['poses'].shape[0]))
print('Vector dmpls has %d elements for each of %d frames.'%(bdata['dmpls'].shape[1], bdata['dmpls'].shape[0]))
print('Vector trams has %d elements for each of %d frames.'%(bdata['trans'].shape[1], bdata['trans'].shape[0]))
print('Vector betas has %d elements constant for the whole sequence.'%bdata['betas'].shape[0])
print('The subject of the mocap sequence is %s.'%bdata['gender'])

# The provided sample data also has the original mocap marker data. In the real AMASS dataset, only markers for the test set are included. For the rest of the subsets you can obtain the marker data from their respective websites.
# In the following we make PyTorch tensors for parameters controlling different part of the body model.
# 
# **Please note how pose indices for different body parts work.**

root_orient = torch.Tensor(bdata['poses'][:, :3]).to(comp_device) # controls the global root orientation
pose_body = torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device) # controls 156 body
pose_hand = torch.Tensor(bdata['poses'][:, 66:]).to(comp_device) # controls the finger articulation
trans = torch.Tensor(bdata['trans'][:, :3]).to(comp_device) # controls the finger articulation
betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(comp_device) # controls the body shape
dmpls = torch.Tensor(bdata['dmpls'][:]).to(comp_device) # controls soft tissue dynamics

betas = torch.zeros_like(betas)
# print(betas)

# Import the required files for viewing out mesh:

import trimesh
from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.mesh.sphere import points_to_spheres
# from notebook_tools import show_image

# imw, imh=1600, 1600
# mv = MeshViewer(width=imw, height=imh, use_offscreen=True)


# ### Visualize betas and pose_body
# Let's see how our body looks like using the pose and body shape parameters.

pose_body_frame = torch.Tensor(bdata['poses'][0:1, 3:66]).to(comp_device) # controls the body
pose_body_frame_zeros = torch.zeros_like(pose_body_frame)
body = bm(pose_body=pose_body_frame_zeros, betas=betas)
# body_mesh = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
# mv.set_static_meshes([body_mesh])
# # body_image = mv.render(render_wireframe=False)
# body_image = mv.render()
# show_image(body_image)

num_joints = pose_body_frame_zeros.shape[1]//3 + 1
base_position = body.Jtr.detach().numpy()[0, 0:num_joints]
parents = bm.kintree_table[0].long()[:num_joints]
link_lengths = []
for i in range(num_joints):
    if i == 0:
        link_lengths.append(base_position[i] - base_position[i])
    else:
        link_lengths.append(base_position[i] - base_position[parents[i]])

def get_dfs_order(parents):
    parents_np = parents.detach().numpy()
    stack = []
    def dfs(stack, joint):
        stack.append(joint)
        for i in range(len(parents_np)):
            if parents_np[i] == joint:
                dfs(stack, i)
    dfs(stack, 0)
    return stack

def create_skeleton_from_amass_bodymodel(bm, betas, num_joints, joint_names, vup='y'):
    pose_body_zeros = torch.zeros((1, 3*(num_joints)-1))
    body = bm(pose_body=pose_body_frame_zeros, betas=betas)
    base_position = body.Jtr.detach().numpy()[0, 0:num_joints]
    parents = bm.kintree_table[0].long()[:num_joints]
    
    joints = []
    for i in range(num_joints):
        joint = kinematics.Joint(name=joint_names[i])
        if i==0:
            joint.info['dof'] = 6
            joint.xform_from_parent_joint = mmMath.p2T(np.zeros(3))
        else:
            joint.info['dof'] = 3
            joint.xform_from_parent_joint = mmMath.p2T(base_position[i]-base_position[parents[i]])        
        joints.append(joint)
    
    parent_joints = []
    for i in range(num_joints):
        parent_joint = None if parents[i]<0 else joints[parents[i]]
        parent_joints.append(parent_joint)

    skel = kinematics.Skeleton(vup=vup)
    for i in range(num_joints):
        skel.add_joint(joints[i], parent_joints[i])
    skel.detect_end_effectors()

    # motion = kinematics.Motion(skel=skel)
    # t = 0.0
    # for i in range(5):
    #     pose_data = [mmMath.I_SE3() for _ in range(num_joints)]
    #     motion.add_one_frame(t, pose_data)
    #     t += 1/30.0
    # motion.save_bvh('data/temp/amass_hierarchy_male.bvh')

    return skel

def create_motion_from_amass_data(filename, skel, bm, betas, dfs_joint_order, fps, fix_height=True, margin_height=0.05, env_vup='z'):
    bdata = np.load(filename)

    fps = float(bdata['mocap_framerate'])
    root_orient = bdata['poses'][:, :3] # controls the global root orientation
    pose_body = bdata['poses'][:, 3:66] # controls 156 body
    pose_hand = bdata['poses'][:, 66:] # controls the finger articulation
    trans = bdata['trans'][:, :3] # controls the finger articulation

    num_joints = skel.num_joint()
    parents = bm.kintree_table[0].long()[:num_joints]

    motion = kinematics.Motion(skel=skel)

    for frame in range(pose_body.shape[0]):
        pose_body_frame = pose_body[frame]
        root_orient_frame = root_orient[frame]
        root_trans_frame = trans[frame]
        pose_data = []
        for j in dfs_joint_order:
            if j == 0:
                T = mmMath.Rp2T(mmMath.exp(root_orient_frame), root_trans_frame)
            else:
                T = mmMath.R2T(mmMath.exp(pose_body_frame[(j-1)*3:(j-1)*3+3]))
            # T = mmMath.I_SE3()
            pose_data.append(T)
        motion.add_one_frame(frame/fps, pose_data)

    if fix_height:
        if env_vup=='y':
            h_idx = 1
        elif env_vup=='z':
            h_idx = 2
        else:
            raise NotImplementedError
        hs = []
        for i in range(motion.num_frame()):
            pose = motion.get_pose_by_frame(i)
            for j in range(num_joints):
                p = mmMath.T2p(pose.get_transform(j, local=False))
                hs.append(p[h_idx])
        # h_min = np.min(h_min)
        if len(hs) > 5:
            h_min = np.mean(nsmallest(5, hs))
        else:
            h_min = np.min(hs)
        h_pen = margin_height - h_min
        h_offset = np.zeros(3)
        h_offset[h_idx] = h_pen
        motion.translate(h_offset)

    motion.resample(fps=fps)

    return motion

def create_motions(job_idx):
    global bm, betas, num_joints, dfs_joint_order, skel
    
    res = []
    num_jobs = job_idx[1] - job_idx[0]
    if num_jobs <= 0: return res
    
    cnt = 0
    time_checker = basics.TimeChecker()
    while cnt < num_jobs:
        file_in = mp.shared_data[job_idx[0]+cnt]
        print('[%d]: %s'%(mp.get_pid(), file_in))
        pre, ext = os.path.splitext(file_in)
        m = create_motion_from_amass_data(filename=file_in,
                                          skel=skel,
                                          bm=bm,
                                          betas=betas,
                                          dfs_joint_order=dfs_joint_order,
                                          fps=30,
                                          )
        ''' Save motions as BVH '''
        m.save_bvh("%s.bvh"%(pre))
        # ''' Save motions as Binary '''
        # with gzip.open("%s.gzip"%(pre), "wb") as f:
        #     pickle.dump(m, f)
        cnt += 1
        if int(0.1*num_jobs) > 0 and cnt % int(0.1*num_jobs)==0:
            percentage = 10*(cnt)//int(0.1*num_jobs)
            time_elapsed = time_checker.get_time(restart=True)
            print('[%d] %d %% completed / %f sec'%(mp.get_pid(),percentage,time_elapsed))
    return res

def get_file_list(target_dir, ext, excludes=['shape.npz'], include_ext=True, sort=True):
    files = []
    for d in target_dir:
        files_cur = basics.files_in_dir(d, ext)
        files_filtered = []
        for f in files_cur:
            add = True
            for e in excludes:
                if e in f:
                    add = False
                    break
            if add:
                if not include_ext: f, _ = os.path.splitext(f)
                files_filtered.append(f)
        files += files_filtered
    if sort: files.sort()
    return files

def auto_labels(job_idx, h_min_max=1.0, h_min_thres=0.2, duration=1.0):
    res = []
    num_jobs = job_idx[1] - job_idx[0]
    if num_jobs <= 0: return res

    labels = []

    def get_heights(pose):
        heights = []
        for j in pose.skel.joints:
            p = mmMath.T2p(pose.get_transform(j, local=False))
            p = mmMath.projectionOnVector(p, pose.skel.v_up_env)
            heights.append(np.linalg.norm(p))
        return heights

    for i in range(job_idx[0], job_idx[1]):
        m = mp.shared_data[i]
        t_remaining = duration
        t_prev = m.times[0]
        use = True
        # print('=================================')
        # print(files[i])
        # print('=================================')
        for frame in range(m.num_frame()):
            pose = m.get_pose_by_frame(frame)
            heights = get_heights(pose)
            idx = np.argpartition(heights, 2)[:2]
            h_min = np.mean(np.array(heights)[idx])
            # print(h_min)
            ''' 
            If the minimum height is too high, there should be some stairs 
            '''
            if h_min >= h_min_max: 
                use = False
                break
            ''' 
            If the minimum height is above then h_min_thres for duration, 
            there should be some stairs 
            '''
            if frame > 0:
                if h_min >= h_min_thres:
                    t_remaining -= (m.times[frame] - t_prev)
                else:
                    t_remaining = duration
                if t_remaining <= 0:
                    use = False
                    break
            t_prev = m.times[frame]
        labels.append(use)

    return labels

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', action='append', default=[])
    parser.add_argument('--dirs', action='append', type=str, default=[])
    parser.add_argument('--num_worekr', type=int, default=1)

if __name__ == '__main__':

    args = arg_parser().parse_args()

    ''' BUGS: If using skel directely, I got some error on kinematics computation '''
    # skel = create_skeleton_from_amass_bodymodel(bm, betas, num_joints, joint_names)

    dfs_joint_order = get_dfs_order(parents)

    motion = kinematics.Motion(file='data/motion/amass/amass_hierarchy.bvh', 
                               v_up_skel=np.array([0.0, 1.0, 0.0]),
                               v_face_skel=np.array([0.0, 0.0, 1.0]),
                               v_up_env=np.array([0.0, 0.0, 1.0]),
                               )
    skel = motion.skel

    files = args.files
    for d in args.dirs:
        files += get_file_list(args.file_dir, ".npz", sort=False)
        files = np.random.permutation(files)

    assert len(files) > 0, 'No file to process'

    ''' Checking whether labels are fine '''
    # with open('data/motion/amass/CMU/label.txt', 'r') as f:
    #     check = np.zeros(len(files))
    #     for line in f:
    #         l = re.split('[\t|\n| ]+', line)
    #         found = False
    #         key = '%s/%s'%(l[0],l[1])
    #         for i, motion_file in enumerate(files):
    #             if key in motion_file:
    #                 found = True
    #                 check[i] = 1.0
    #                 break
    #         if not found:
    #             print(line)
    #     for i in range(len(files)):
    #         if check[i] == 0.0:
    #             print(i, files[i])

    # print(len(files))
    # exit(0)

    mp.shared_data = files
    motions = mp.run_parallel_async_idx(create_motions,
                                        args.num_worker, 
                                        len(mp.shared_data),
                                        )

    # import amass_char_info as char_info
    # import create_motiongraph as cmg

    # files = get_file_list(target_dir, ".bvh", sort=True)
    # mp.shared_data = files
    # motions = mp.run_parallel_async_idx(cmg.read_motions, 
    #                                     args.num_worker, 
    #                                     len(mp.shared_data), 
    #                                     skel, 
    #                                     1.0, 
    #                                     char_info.v_up,
    #                                     char_info.v_face,
    #                                     char_info.v_up_env,
    #                                     )

    # mp.shared_data = motions
    # labels = mp.run_parallel_async_idx(auto_labels,
    #                                    num_worker, 
    #                                    len(mp.shared_data),
    #                                    )

    # for i in range(len(labels)):
    #     print(labels[i], files[i])


