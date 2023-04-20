# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for Uniformly Random Global Pruning (without any rescaling)'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The probability of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    # def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
    def prune(pruning_hparams: PruningHparams,
              trained_model: models.base.Model,
              current_mask: Mask = None,
              training_hparams: hparams.TrainingHparams = None,
              dataset_hparams: hparams.DatasetHparams = None,
              data_order_seed: int = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Get the model weights.
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_tensors}

        pruning_fraction = Strategy.get_pruning_hparams().pruning_fraction
        new_mask = Mask({k: np.where(current_mask[k], np.random.choice([0,1], v.shape, p=[pruning_fraction, 1 - pruning_fraction]), np.zeros_like(v))
                         for k, v in weights.items()})
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
