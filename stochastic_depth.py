import torch
import torch.nn as nn
from torch import Tensor

def stochastic_depth(inputs: Tensor, p: float, training: bool):
    assert p >= 0.0 and p <= 1.0, f"p should be in the range of [0, 1] but found {p}"
    # if p is zero (survivial rate is one) or in eval mode
    if p == 0.0 or not training:
        return inputs

    batch_size = inputs.shape[0]
    size = [batch_size] + [1]*(inputs.dim()-1) # this will create a per-data-example mask, which mask out an entire data example or not
    mask = torch.empty(size, dtype=inputs.dtype, device=inputs.device)
    survival_rate = 1.0 - p
    mask.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        mask.div_(survival_rate)
    return inputs * mask



class StochasticDepth(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        
    def forward(self, inputs):
        return stochastic_depth(inputs, self.p, self.training)

    def __repr__(self):
        return f'p={self.p}'

# we can draw resemblance between stochastic_depth and dropout
# dropout is applied to neurons inside a layer
# stochastic depth can be seen as dropout applied to the granularity of layers inside a net
def dropout(inputs: Tensor, p:float, training:bool):
    assert p >= 0.0 and p <= 1.0, f"p should ve in the range of [0, 1] but found {p}"
    # if p is zero (survivial rate is one) or in eval mode
    if p == 0.0 or not training:
        return inputs

    mask = torch.new_empty(input)
    survival_rate = p
    mask.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        mask.div_(survival_rate)
    return inputs * mask

class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    def forward(self, inputs):
        return dropout(inputs, self.p, self.training)