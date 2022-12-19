import os
import numpy as np

from fairmotion.core import motion as mo_core
from fairmotion.ops import motion as mo_ops

from fairmotion.ops import conversions
from fairmotion.ops import math

from fairmotion.data import bvh
import copy

info = {
    "motions": [
        # {
        #     "file": "data/motion/multiagent/boxing/base/3_punch__poses.bvh",
        #     "save_as": "data/motion/multiagent/boxing/base/3_punch__poses_reflected.bvh",
        #     "description": None,
        # },
        # {
        #     "file": "data/motion/multiagent/boxing/base/avoid_poses.bvh",
        #     "save_as": "data/motion/multiagent/boxing/base/avoid_poses_reflected.bvh",
        #     "description": None,
        # },
        # {
        #     "file": "data/motion/multiagent/boxing/base/idle_poses.bvh",
        #     "save_as": "data/motion/multiagent/boxing/base/idle_poses_reflected.bvh",
        #     "description": None,
        # },
        # {
        #     "file": "data/motion/multiagent/boxing/base/jab001_poses.bvh",
        #     "save_as": "data/motion/multiagent/boxing/base/jab001_poses_reflected.bvh",
        #     "description": None,
        # },
        {
            "file": "data/motion/multiagent/boxing/experts/avoid_guard.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/avoid_guard_reflected.bvh",
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/experts/avoid_left_large.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/avoid_left_large_reflected.bvh",
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/experts/avoid_left_small.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/avoid_left_small_reflected.bvh",
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/experts/avoid_right_large.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/avoid_right_large_reflected.bvh",
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/experts/avoid_right_small.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/avoid_right_small_reflected.bvh",
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/experts/idle_fw_fake.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/idle_fw_fake_reflected.bvh",
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/experts/idle_left.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/idle_left_reflected.bvh",
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/experts/idle_right.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/idle_right_reflected.bvh",
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/experts/idle_stand.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/idle_stand_reflected.bvh",
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/experts/jab1.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/jab1_reflected.bvh",
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/experts/punch1.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/punch1_reflected.bvh",
            "description": None,
        },
    ],
    "joint_pair": {
        "root": {
            "pair": "root",
            "reflection_plane": "yz"
        },

        "lhip": {
            "pair": "rhip",
            "reflection_plane": "yz"
        },
        "lknee": {
            "pair": "rknee",
            "reflection_plane": "yz"
        },
        "lankle": {
            "pair": "rankle",
            "reflection_plane": "yz"
        },
        "ltoe": {
            "pair": "rtoe",
            "reflection_plane": "yz"
        },
        "rhip": {
            "pair": "lhip",
            "reflection_plane": "yz"
        },
        "rknee": {
            "pair": "lknee",
            "reflection_plane": "yz"
        },
        "rankle": {
            "pair": "lankle",
            "reflection_plane": "yz"
        },
        "rtoe": {
            "pair": "ltoe",
            "reflection_plane": "yz"
        },


        "lshoulder": {
            "pair": "rshoulder",
            "reflection_plane": "yz"
        },
        "lelbow": {
            "pair": "relbow",
            "reflection_plane": "yz"
        },
        "lwrist": {
            "pair": "rwrist",
            "reflection_plane": "yz"
        },
        "rshoulder": {
            "pair": "lshoulder",
            "reflection_plane": "yz"
        },
        "relbow": {
            "pair": "lelbow",
            "reflection_plane": "yz"
        },
        "rwrist": {
            "pair": "lwrist",
            "reflection_plane": "yz"
        },


        "lowerback": {
            "pair": "lowerback",
            "reflection_plane": "yz"
        },
        "upperback": {
            "pair": "upperback",
            "reflection_plane": "yz"
        },
        "chest": {
            "pair": "chest",
            "reflection_plane": "yz"
        },
        "lowerneck": {
            "pair": "lowerneck",
            "reflection_plane": "yz"
        },
        "upperneck": {
            "pair": "upperneck",
            "reflection_plane": "yz"
        },

        "lclavicle": {
            "pair": "rclavicle",
            "reflection_plane": "yz"
        },
        "lshoulder": {
            "pair": "rshoulder",
            "reflection_plane": "yz"
        },
        "lelbow": {
            "pair": "relbow",
            "reflection_plane": "yz"
        },
        "lwrist": {
            "pair": "rwrist",
            "reflection_plane": "yz"
        },
        "rclavicle": {
            "pair": "lclavicle",
            "reflection_plane": "yz"
        },
        "rshoulder": {
            "pair": "lshoulder",
            "reflection_plane": "yz"
        },
        "relbow": {
            "pair": "lelbow",
            "reflection_plane": "yz"
        },
        "rwrist": {
            "pair": "lwrist",
            "reflection_plane": "yz"
        },
    },
}

if __name__ == '__main__':

    motions = info["motions"]
    joint_pair = info["joint_pair"]
    loaded_motions = {}
    
    for d in motions:
        
        if d["file"] in loaded_motions.keys():
            m = loaded_motions[d["file"]]
        else:
            if not os.path.isfile(d["file"]):
                print("File does not exist", d["file"])
                continue
            m = bvh.load(
                file=d["file"],
                scale=1.0,
                v_up_skel=np.array([0.0, 1.0, 0.0]),
                v_face_skel=np.array([0.0, 0.0, 1.0]),
                v_up_env=np.array([0.0, 0.0, 1.0])
            )
            loaded_motions[d["file"]] = m

        m_reflected = copy.deepcopy(m)

        for i in range(m.num_frames()):
            pose = m.get_pose_by_frame(i)
            pose_reflected = m_reflected.get_pose_by_frame(i)
            for j in joint_pair.keys():
                j_pair = joint_pair[j]["pair"]
                reflection_plane = joint_pair[j]["reflection_plane"]
                R, p = conversions.T2Rp(pose.get_transform(j_pair, local=True))
                if reflection_plane == "yz":
                    y = R[:,1].copy()
                    z = R[:,2].copy()
                    y[0] *= -1
                    z[0] *= -1
                    x = np.cross(y,z)
                    R_reflected = np.array([x, y, z]).transpose()
                    p_reflected = np.array([-p[0], p[1], p[2]])
                    pose_reflected.set_transform(
                        j, 
                        conversions.Rp2T(R_reflected, p_reflected),
                        local=True)

        save_file_name = d["save_as"]
        bvh.save(m_reflected, save_file_name)

        print("%s -> %s" % (d["file"], save_file_name))
