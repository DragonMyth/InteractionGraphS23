import numpy as np
import torch

from . import featurizer
from fairmotion.tasks.motion_plausibility import (
    featurizer as mp_featurizer,
    model as mp_model,
    test,
)


class ScorerFactory:
    def get_scorer(
        path, model_type="mlp", feature_type="rotation", rep="aa", device="cpu"
    ):
        if model_type == "mlp":
            return PlausibilityScorer(
                path=path,
                feature_type=feature_type,
                device=device,
            )
        else:
            return NearestNeighborPlausibilityScorer(
            path=path,
            feature_type=feature_type,
        )


class PlausibilityScorer:
    def __init__(self, path, feature_type="rotation", device="cpu", rep="aa"):
        """Model to score plausibility of pose given observed history of poses.

        Args:
            path: str; Path to stored motion plausibility model
            feature_type: ["rotation", "facing"]; Type of featurizer to use
            device: ["cpu", "cuda]; Device on which model is run
        """
        self.model, self.model_kwargs, stats = test.load_model(path, device)
        mean, std = stats
        self.num_observed = self.model_kwargs["num_observed"]
        if feature_type == "rotation":
            self.featurizer = mp_featurizer.RotationFeaturizer(
                rep=rep, mean=mean, std=std,
            )
        else:
            self.featurizer = mp_featurizer.FacingPositionFeaturizer(
                mean=mean, std=std,
            )
        self.device = device

    def evaluate(self, prev_poses, cur_pose):
        """
        prev_poses: List of previous observed Pose objects. Expected size is
            (num_observed,)
        cur_pose: Curent Pose object
        """
        assert len(prev_poses) == self.num_observed, (
            f"The number of prev_poses ({len(prev_poses)}) is not the same as "
            "the number of prev_poses ({self.num_observed}) that the "
            "plausibility model is trained on"
        )
        prev_data, cur_data = self.featurizer.featurize(prev_poses, cur_pose)
        prev_poses = torch.Tensor(prev_data).double().to(self.device)
        cur_pose = torch.Tensor(cur_data).double().to(self.device)
        score = self.model(prev_poses, cur_pose)
        return score.detach().cpu().numpy()[0][0]


class ZScorer:
    def __init__(self, model_path, device="cpu"):
        self.model, self.model_kwargs, model_stats = test.load_model(
            model_path, device,
        )
        self.num_observed = self.model_kwargs["num_observed"]
        self.featurizer = featurizer.ZFeaturizer(
            mean=model_stats[0], std=model_stats[1],
        )
        self.device = device

    def validate_input(self, z_prev, z):
        assert z_prev.shape[-1] == z.shape[-1]
        assert z_prev.shape[-2] == self.num_observed
        if z_prev.ndim == 2:
            z_prev = np.expand_dims(z_prev, 0)
        if z.ndim == 1:
            z = np.expand_dims(z, 0)
        return z_prev, z

    def evaluate(self, z_prev, z):
        z_prev, z = self.validate_input(z_prev, z)
        z_prev, z = self.featurizer.featurize(z_prev, z)
        z_prev = torch.Tensor(z_prev).double().to(self.device)
        z = torch.Tensor(z).double().to(self.device)
        score = self.model(z_prev, z)
        return score.detach().cpu().numpy()[0][0]


class NearestNeighborPlausibilityScorer:
    def __init__(self, path, feature_type="rotation", rep="aa"):
        """Nearest Neighbor model to score plausibility of pose.

        Args:
            path: str; Path to stored motion plausibility model
            feature_type: ["rotation", "facing"]; Type of featurizer to use
        """
        if feature_type == "rotation":
            self.featurizer = mp_featurizer.RotationFeaturizer(rep=rep)
        else:
            self.featurizer = mp_featurizer.FacingPositionFeaturizer()
        self.model = mp_model.NearestNeighbor.load(path)
        self.num_observed = self.model.num_observed


    def evaluate(self, prev_poses, cur_pose):
        """
        prev_poses: List of previous observed Pose objects. Expected size is
            (num_observed,)
        cur_pose: Curent Pose object
        """
        prev_poses.append(cur_pose)
        return self.model.score(prev_poses)
