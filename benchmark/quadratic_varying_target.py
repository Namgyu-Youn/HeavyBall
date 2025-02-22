import itertools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import typer

import heavyball
from heavyball.utils import set_torch
from benchmark.utils import param_norm_win_condition, trial

app = typer.Typer(pretty_exceptions_enable=False)
set_torch()

class Model(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.register_buffer('target', F.normalize(torch.arange(size).float().add(1).square(), dim=0))

    def forward(self):
        return (self.param - self.target).square().mean()

@app.command()
def main(
    method: List[str] = typer.Option(['qr'], help='Eigenvector method to use (for SOAP)'),
    dtype: List[str] = typer.Option(['float32'], help='Data type to use'),
    size: int = 1024,
    batch: int = 256,
    steps: int = 100,
    weight_decay: float = 0,
    opt: List[str] = typer.Option(['ForeachSOAP'], help='Optimizers to use'),
    trials: int = 10,
    win_condition_multiplier: float = 1.0,
):
    dtype = [getattr(torch, d) for d in dtype]
    model = Model(size).cuda()

    def data():
        return None, None


    trial(model, data, None, param_norm_win_condition(win_condition_multiplier * 1e-8, -model.target), steps, opt[0], dtype[0], size, batch, weight_decay, method[0], 1, 1,
          failure_threshold=2, base_lr=1e-3, trials=trials)

if __name__ == '__main__':
    app()
