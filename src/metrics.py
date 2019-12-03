import numpy as np
import torch
import os
import math
import scipy.stats
from collections import Counter
from data_loader import get_loader


class FactorVAEMetric(object):
    def __init__(self, config, model, device):
        self.config = config
        self.device = device
        self.model = model
        self.num_eval_global_var = config.num_eval_global_var
        self.data_loader = get_loader(
            config.eval_batch_size, config.project_root, config.dataset)
        self.dset = self.data_loader.dataset
        self._compute_global_variance()

    def _compute_global_variance(self):
        random_images = self.dset.imgs[np.random.choice(
            len(self.dset), self.num_eval_global_var)]
        random_images = torch.from_numpy(random_images).to(self.device)
        representations = self.model(random_images)

        return

    def _compute_variance(self, representations):

        pass

    def evaluate(self, epoch_id, batch_id, global_id):

        pass
