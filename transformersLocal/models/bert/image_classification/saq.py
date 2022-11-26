import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging
import abc
import sys
from .quantize import QLinear
from collections import defaultdict


class SAQ:
    def __init__(
            self,
            optimizer,
            model,
            rho=0.5,
            include_wclip=False,
            include_aclip=False,
            include_bn=True,
    ):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.include_wclip = include_wclip
        self.include_aclip = include_aclip
        self.include_bn = include_bn
        self.state = defaultdict(dict)

    @torch.no_grad()
    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.optimizer.param_groups[0]["params"][0].device
        wgrads = []
        for n, m in self.model.named_modules():
            if isinstance(m, (QLinear)):
                wgrads.append(torch.norm(m.x.grad, p=2).to(shared_device))

                if hasattr(m, "bias") and m.bias is not None:
                    wgrads.append(torch.norm(m.bias.grad, p=2).to(shared_device))

        wgrad_norm = torch.norm(torch.stack(wgrads), p=2)
        return wgrad_norm

    @torch.no_grad()
    def ascent_step(self):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for n, m in self.model.named_modules():
            if isinstance(m, (QLinear)):
                p = m.x
                self.state[m]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                m.epsilon = e_w

                if hasattr(m, "bias") and m.bias is not None:
                    p = m.bias
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)


        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, m in self.model.named_modules():
            if isinstance(m, (QLinear)):
                if hasattr(m, "bias") and m.bias is not None:
                    p = m.bias
                    p.data = self.state[p]["old_p"]

        # for n, m in self.model.named_modules():
        #     if isinstance(m, (QLinear)):
        #         if m.epsilon is None:
        #             m.epsilon = torch.zeros_like(m.x)

        self.optimizer.step()
        self.optimizer.zero_grad()


def set_first_forward(model):
    for n, m in model.named_modules():
        if isinstance(m, QLinear):
            m.set_first_forward()


def set_second_forward(model):
    for n, m in model.named_modules():
        if isinstance(m, QLinear):
            m.set_second_forward()
