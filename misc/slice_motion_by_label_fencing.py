import os
import numpy as np

from fairmotion.core import motion as mo_core
from fairmotion.ops import motion as mo_ops

from fairmotion.data import bvh

label_file = {
    "description": "experts for fencing",
    "data": [
        {
            "file": "data/motion/multiagent/fencing/base/Capture1_subject1_stageII.bvh",
            "save_as": "data/motion/multiagent/fencing/experts/idle.bvh",
            "frame_start": 30,
            "frame_end": 75,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/fencing/base/Capture1_subject1_stageII.bvh",
            "save_as": "data/motion/multiagent/fencing/experts/step_forward.bvh",
            "frame_start": 150,
            "frame_end": 190,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/fencing/base/Capture2_subject1_stageII.bvh",
            "save_as": "data/motion/multiagent/fencing/experts/step_backward.bvh",
            "frame_start": 250,
            "frame_end": 285,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/fencing/base/Capture1_subject1_stageII.bvh",
            "save_as": "data/motion/multiagent/fencing/experts/attack_small.bvh",
            "frame_start": 95,
            "frame_end": 140,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/fencing/base/Capture1_subject1_stageII.bvh",
            "save_as": "data/motion/multiagent/fencing/experts/attack_middle.bvh",
            "frame_start": 190,
            "frame_end": 225,
            "description": None,
        },
        {
            "file": "data/motion/multiagent/fencing/base/Capture1_subject1_stageII.bvh",
            "save_as": "data/motion/multiagent/fencing/experts/attack_large.bvh",
            "frame_start": 982,
            "frame_end": 1040,
            "description": None,
        },
        
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
