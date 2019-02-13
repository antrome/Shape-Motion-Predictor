from __future__ import division

import numpy as np
from src.transforms.transforms import TransformBase
import torch

class Normalize(TransformBase):
    """Normalize tensor with given mean std
    """

    def __init__(self, perkey_args, general_args):
        super(Normalize, self).__init__()
        self._params = self._find_params_for_sample_key(perkey_args, general_args)

    def __call__(self, sample):
        for sample_key in sample.keys():
            if sample_key in self._params.keys():
                elem = sample[sample_key]
                mean = torch.FloatTensor(self._params[sample_key]["mean"])
                std = torch.FloatTensor(self._params[sample_key]["std"])
                sample[sample_key] = self._normalize(elem, mean, std)
        return sample

    def _normalize(self, sample, mean, std):
        return (sample - mean) / std

    def __str__(self):
        return 'Normalize:' + str(self._params)
