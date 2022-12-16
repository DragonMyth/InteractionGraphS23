from fairmotion.tasks.motion_plausibility import featurizer


class ZFeaturizer(featurizer.Featurizer):
    def __init__(self, mean=None, std=None):
        super().__init__(mean, std)

    def featurize(self, z_prev, z):
        return self.normalize(z_prev), self.normalize(z)
