"""Versioned perceptual-feature diagnostic; historical pilot utilities stay frozen."""
from cctv_dgp_pilot import PilotPerceptual


class PerceptualV3(PilotPerceptual):
    """Compare activation placement using exactly the same frozen VGG trunk."""
    def __init__(self, path, device, feature_policy='postactivation'):
        super().__init__(path, device)
        self.select_policy(feature_policy)

    def select_policy(self, feature_policy):
        if feature_policy not in ('postactivation', 'preactivation'):
            raise ValueError('Unknown perceptual feature policy')
        self.feature_policy = feature_policy

    def taps(self, x):
        indices = {'postactivation': (3, 8, 17, 26),
                   'preactivation': (2, 7, 16, 25)}
        if self.feature_policy not in indices:
            raise ValueError('Unknown perceptual feature policy')
        x = (x - self.mean) / self.std
        outputs = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in indices[self.feature_policy]:
                # The next ReLU is in-place. Retain signed values and their
                # gradient path rather than a view that the next layer changes.
                outputs.append(x.clone())
        return outputs
