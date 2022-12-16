from multiprocessing import Pool
import argparse
import copy
import os
import tqdm

from basecode.motion import kinematics_simple as kinematics
import pfnn
from mpi4py import MPI
import numpy as np
import os

def generate_and_save_pfnn(inputs):
    args = inputs[0]
    runner = inputs[1]
    motion = inputs[2]
    index = inputs[3]

    # runner = inputs[0]
    # index = inputs[1]
    
    # Run PFNN for a second to start with arbitrary pose
    for i in range(0):
        runner.update()
    # Clear frames from motion object
    motion.clear()

    while(motion.num_frame() <= 0 or motion.times[-1] < args.clip_time):
    # while(motion.num_frame() <= 0 or motion.times[-1] < 5):
        for i in range(2):
            runner.update()
        if motion.num_frame() > 0:
            t = motion.times[-1]+1.0/30.0
        else:
            t = 0.0
        motion.add_one_frame(t, copy.deepcopy(runner.character.joint_xform_by_ik))

    filename = os.path.join(args.output_folder, f"random_pfnn_{index}.bvh")
    motion.save_bvh(filename, scale=0.01, verbose=True)
    print('Saved:', filename)

def main(args):
    runner = pfnn.Runner(user_input='autonomous')
    motion = kinematics.Motion(file="data/motion/pfnn/pfnn_hierarchy.bvh", scale=1.0)
    # runner_copy = copy.deepcopy(runner)
    # pool = Pool(args.cpus)
    # pool.map(generate_and_save_pfnn, zip([runner_copy] * args.num_clips, range(args.num_clips)))
    
    # for i in tqdm.tqdm(range(args.num_clips)):
    #     generate_and_save_pfnn([args, runner, motion, i])

    np.random.seed(os.getpid())

    rank = MPI.COMM_WORLD.Get_rank()
    for i in range(args.num_clips):
        generate_and_save_pfnn([args, runner, motion, rank*args.num_clips+i])


def translate_bvh_to_origin(filepath):
    motion = kinematics.Motion(file=filepath,
                               v_up_skel=np.array([0.0, 1.0, 0.0]),
                               v_face_skel=np.array([0.0, 0.0, 1.0]),
                               v_up_env=np.array([0.0, 1.0, 0.0]))
    position = motion.postures[0].get_facing_position()
    translate_by = [-1 * position[0], 0 , -1 * position[2]]
    motion.translate(translate_by)
    motion.save_bvh(f"{filepath.split('.')[0]}_translated.bvh")


def translate_bvh_to_origin_all(args):
    p = Pool(args.cpus)
    for root, _, files in os.walk(args.output_folder, topdown=False):
        p.map(translate_bvh_to_origin, [os.path.join(root, filename) for filename in files])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate arbitrary action sequences from PFNN and save them to BVH")
    parser.add_argument(
        "--clip-time", type=int, help="Time of each clip (in seconds)", default=5,
    )
    parser.add_argument(
        "--num-clips", type=int, help="Number of clips to be generated",
    )
    parser.add_argument(
        "--output-folder", type=str, help="Output folder where generated BVH files are stored", required=True,
    )
    parser.add_argument(
        "--cpus", type=int, help="Number of CPUs for multiprocessing", default=10, required=False,
    )
    parser.add_argument(
        "--task", type=str, choices=["translate", "generate"], help="Defines task to be performed", default="generate"
    )
    args = parser.parse_args()
    if args.task == "translate":
        translate_bvh_to_origin_all(args)
    else:
        main(args)
