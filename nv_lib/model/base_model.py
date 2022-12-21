import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def extra_state_dict(self):
        return {}

    def load_extra_state_dict(self, state_dict):
        pass
