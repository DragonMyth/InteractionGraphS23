import argparse
import numpy as np
import os
import pickle

from . import options
from fairmotion.tasks.motion_plausibility import options as fairmotion_options
from fairmotion.utils import utils as fairmotion_utils


def get_negative_samples(shape=(4)):
    return np.random.uniform(size=shape, low=-1, high=1)


def preprocess_episode(z, num_observed=5, skip_frames=1, stride=1):
    selected_data = []
    indices = np.arange(num_observed + 1) * skip_frames
    for i in range(0, len(z) - (num_observed + 1) * skip_frames, stride):
        selected_data.append(np.take(z, i + indices, axis=0))
    selected_data = np.array(selected_data)
    return selected_data


def create_splits(data, shuffle=True, train_ratio=0.8):
    if shuffle:
        np.random.shuffle(data)
    size = len(data)
    return data[:int(size*train_ratio)], data[int(size*train_ratio):]


def append_negative_samples(data, negative_sample_ratio=5):
    num_samples = len(data)
    data = np.array(list(data) * (negative_sample_ratio + 1))
    negative_samples = get_negative_samples(
        shape=(negative_sample_ratio*num_samples, 4),
    )
    data[num_samples:, args.num_observed] = negative_samples
    labels = [1] * num_samples + [0] * negative_sample_ratio*num_samples
    return data, labels


def preprocess_z(args):
    data = []
    episodes = pickle.load(open(args.z_input_path, "rb"))
    for episode in episodes:
        data.extend(
            preprocess_episode(
                z=np.array(episode["z_task"]),
                num_observed=args.num_observed,
                skip_frames=args.frames_between_poses,
                stride=args.stride,
            )
        )
    print(f"{len(data)} samples found")
    print(f"Data shape: {np.array(data).shape}")
    train_data, valid_data = create_splits(
        np.array(data),
        shuffle=True,
        train_ratio=0.8,
    )

    train_data, train_labels = append_negative_samples(train_data)
    valid_data, valid_labels = append_negative_samples(valid_data)
    metadata = {
        "num_observed": args.num_observed,
        "skip_frames": args.frames_between_poses,
    }
    train_dataset = train_data, train_labels, metadata
    valid_dataset = valid_data, valid_labels, metadata

    fairmotion_utils.create_dir_if_absent(args.output_dir)
    train_path = os.path.join(
        args.output_dir,
        (
            f"train_num_observed_{args.num_observed}_skip_frames_"
            f"{args.frames_between_poses}.pkl"
        )
    )
    pickle.dump(train_dataset, open(train_path, "wb"))

    valid_path = os.path.join(
        args.output_dir,
        (
            f"valid_num_observed_{args.num_observed}_skip_frames_"
            f"{args.frames_between_poses}.pkl"
        )
    )
    pickle.dump(valid_dataset, open(valid_path, "wb"))
    pickle.dump(train_dataset, open(train_path, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    fairmotion_options.add_preprocess_args(parser)
    options.add_preprocess_z_args(parser)
    args = parser.parse_args()
    fairmotion_utils.create_dir_if_absent(args.output_dir)
    preprocess_z(args)
