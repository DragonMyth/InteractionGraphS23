import argparse
import itertools
import numpy as np
import os

from collections import defaultdict

from motion_plausibility import scorer
from fairmotion.data import bvh
from fairmotion.utils import utils as fairmotion_utils


def plot(
    bvh_file, model_path, feature_type, skip_frames, plausibility_type, device
):
    model = scorer.ScorerFactory.get_scorer(
        path=model_path,
        model_type=plausibility_type,
        feature_type=feature_type,
        device=device,
    )
    motion = bvh.load(bvh_file)
    data = defaultdict(float)

    indices = np.flip(np.arange(model.num_observed)) * skip_frames
    for idx in range(
        (model.num_observed + 1) * skip_frames,
        motion.num_frames(),
    ):
        prev_poses = []
        for j in (idx - skip_frames) - indices:
            prev_poses.append(motion.poses[j])
        score = model.evaluate(
            prev_poses,
            motion.poses[idx],
        )
        data[idx] = score
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh-files", nargs="+", type=str, required=True)
    parser.add_argument("--model-paths", nargs="+", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--skip-frames", nargs="+", type=int, required=True)
    parser.add_argument(
        "--feature-types", nargs="+", type=str, choices=["facing", "rotation"],
        required=True,
    )
    parser.add_argument(
        "--plausibility-type", nargs="+", type=str, choices=["mlp", "nn"],
        required=True,
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"],
        default="cuda",
    )
    args = parser.parse_args()
    fairmotion_utils.create_dir_if_absent(os.path.dirname(args.output_file))
    data = {}

    for bvh_file, model_path, plausibility_type in itertools.product(
        args.bvh_files, args.model_paths, args.plausibility_type,
    ):
        model_idx = args.model_paths.index(model_path)
        skip_frames = args.skip_frames[model_idx]
        feature_type = args.feature_types[model_idx]

        filename = bvh_file.split("/")[-1].split(".")[0]
        model_name = str("__".join(model_path.split("/")[-3:-1]))
        title = f"{filename}_{feature_type}_{skip_frames}_{model_name}"
        data[title] = plot(
            bvh_file,
            model_path,
            feature_type,
            skip_frames,
            plausibility_type,
            args.device,
        )
    
    frame_numbers = [x for key in data.keys() for x in list(data[key].keys())]
    min_row_idx, max_row_idx = min(frame_numbers), max(frame_numbers)
    
    with open(args.output_file, "w") as file:
        file.write(",".join(["index"] + list(data.keys())) + "\n")
        for idx in range(min_row_idx, max_row_idx + 1):
            row_items = [f"{idx}"]
            for title in data.keys():
                row_items.append(str(data[title][idx]))
            file.write(f"{','.join(row_items)}\n")      
        

    
