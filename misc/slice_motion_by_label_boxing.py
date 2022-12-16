import os
import numpy as np

from fairmotion.core import motion as mo_core
from fairmotion.ops import motion as mo_ops

from fairmotion.data import bvh

label_file = {
    "description": "experts for boxing",
    "data": [
        {
            "file": "data/motion/multiagent/boxing/base/3_punch__poses.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/punch1.bvh",
            "frame_start": 20,
            "frame_end": 100,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/base/avoid_poses.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/avoid_left_small.bvh",
            "frame_start": 0,
            "frame_end": 65,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/base/avoid_poses.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/avoid_right_small.bvh",
            "frame_start": 60,
            "frame_end": 110,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/base/avoid_poses.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/avoid_left_large.bvh",
            "frame_start": 110,
            "frame_end": 155,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/base/avoid_poses.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/avoid_right_large.bvh",
            "frame_start": 180,
            "frame_end": 225,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/base/avoid_poses.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/avoid_guard.bvh",
            "frame_start": 540,
            "frame_end": 570,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/base/idle_poses.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/idle_stand.bvh",
            "frame_start": 225,
            "frame_end": 250,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/base/idle_poses.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/idle_left.bvh",
            "frame_start": 290,
            "frame_end": 340,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/base/idle_poses.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/idle_right.bvh",
            "frame_start": 360,
            "frame_end": 430,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/base/idle_poses.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/idle_fw_fake.bvh",
            "frame_start": 450,
            "frame_end": 500,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/boxing/base/jab001_poses.bvh",
            "save_as": "data/motion/multiagent/boxing/experts/jab1.bvh",
            "frame_start": 185,
            "frame_end": 260,
            "description": None,
        }
    ]
}

if __name__ == '__main__':

    label_data = label_file["data"]
    loaded_motions = {}
    
    for d in label_data:
        
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

        fs, fe = d["frame_start"], d["frame_end"]

        assert 0 <= fs <= m.num_frames()
        assert 0 <= fe <= m.num_frames()
        assert fs <= fe

        m_cut = mo_ops.cut(
            motion=m,
            frame_start=fs,
            frame_end=fe,
        )

        save_file_name = d["save_as"]

        if save_file_name is None:
            name, ext = os.path.splitext(d["file"])
            save_file_name = name + "_(%d_%d).bvh"%(fs, fe)

        bvh.save(m_cut, save_file_name)

        print("%s (%d,%d) -> %s" % (d["file"], fs, fe, save_file_name))
