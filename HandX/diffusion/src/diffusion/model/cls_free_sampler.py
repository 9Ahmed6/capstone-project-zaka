import torch
import torch.nn as nn
from copy import deepcopy

class ClassifierFreeSampleWrapper(nn.Module):
    def __init__(self, model, scale=None):
        super().__init__()
        self.model = model

        # Handle DataParallel: access attributes through .module if wrapped
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        assert base_model.cond_mode != 'no_cond', "ClassifierFreeSampleWrapper only supports models with conditional mode."

        self.scale = scale
        self.cond_mode = base_model.cond_mode
        assert self.cond_mode in ['text', 'action'], f"Unsupported cond_mode: {self.cond_mode}"

    def forward(self, x, timesteps, y=None, *args, **kwargs):
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        uncond_output = self.model(x, timesteps, y_uncond, *args, **kwargs)
        output = self.model(x, timesteps, y, *args, **kwargs)

        if not 'scale' in y.keys():
            y['scale'] = torch.ones(output.shape[0], device=x.device) * self.scale

        output_dim = len(output.shape) - 1
        target_shape = (-1,) + (1,) * output_dim

        return uncond_output + (y['scale'].view(*target_shape) * (output - uncond_output))
