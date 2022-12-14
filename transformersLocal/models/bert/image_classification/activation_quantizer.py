# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import logging
import math

class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        ctx.num_bits = num_bits
        if num_bits < 32:
            input = torch.where(input < clip_val[1], input, clip_val[1])
            input = torch.where(input > clip_val[0], input, clip_val[0])
            # NOTE: dynamic scaling (max_input).
            if layerwise == 'layer':
                max_input = torch.max(torch.abs(input)).expand_as(input)
            elif layerwise == 'row':
                if input.ndimension() <= 3:
                    # weight & hidden layer
                    max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
                elif input.ndimension() == 4:
                    # TODO: attention score matrix, calculate alpha / beta per head
                    tmp = input.view(input.shape[0], input.shape[1], -1)
                    max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(
                        input).detach()
                else:
                    raise ValueError
            s = (2 ** (num_bits - 1) - 1) / max_input
            output = torch.round(input * s).div(s)
        else:
            output = input

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        num_bits = ctx.num_bits
        grad_input = grad_output.clone()
        grad_clip = None
        if num_bits < 32:
            grad_input[input.ge(clip_val[1])] = 0
            grad_input[input.le(clip_val[0])] = 0
            # refer to PACT
            grad_clip_pos = (grad_output * input.ge(clip_val[1]).float()).sum()
            grad_clip_neg = (grad_output * (input.le(clip_val[0]).float())).sum()
            grad_clip = torch.tensor([grad_clip_neg, grad_clip_pos]).to(input.device)
        return grad_input, grad_clip, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        ctx.num_bits = num_bits
        if num_bits < 32:
            input = torch.where(input < clip_val[1], input, clip_val[1])
            input = torch.where(input > clip_val[0], input, clip_val[0])
            # NOTE: dynamic scaling gives better performance than static
            if layerwise == 'layer':
                alpha = (input.max() - input.min()).detach()
                beta = input.min().detach()
                # alpha = clip_val[1] - clip_val[0]
                # beta = clip_val[0]
            elif layerwise == 'row':
                if input.ndimension() <= 3:
                    # weight & hidden layer
                    alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(
                        input).detach()
                    beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
                elif input.ndimension() == 4:
                    # TODO: attention score matrix, calculate alpha / beta per head
                    tmp = input.view(input.shape[0], input.shape[1], -1)
                    alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                             tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                    beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
                else:
                    raise ValueError
            input_normalized = (input - beta) / (alpha + 1e-8)
            s = (2 ** num_bits - 1)
            quant_input = torch.round(input_normalized * s).div(s)
            output = quant_input * (alpha + 1e-8) + beta
        else:
            output = input

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        num_bits = ctx.num_bits
        grad_input = grad_output.clone()
        grad_clip = None
        if num_bits < 32:
            grad_input[input.ge(clip_val[1])] = 0
            grad_input[input.le(clip_val[0])] = 0
            # refer to PACT
            grad_clip_pos = (grad_output * input.ge(clip_val[1]).float()).sum()
            grad_clip_neg = (grad_output * (input.le(clip_val[0]).float())).sum()
            grad_clip = torch.tensor([grad_clip_neg, grad_clip_pos]).to(input.device)
        return grad_input, grad_clip, None, None


class SymLsqQuantizer(torch.autograd.Function):
    """
        Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise, learnable):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1

        assert alpha.min() > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)
        if alpha.view(-1)[0] == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=True, init_method='default', layerwise=layerwise)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        # grad_scale = 1.0
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, learnable, layerwise

        if layerwise == 'layer':
            q_w = (input / alpha).round().clamp(Qn, Qp)
        elif layerwise == 'row':
            # print("183 input {} alpha {} ".format(input.shape, alpha.shape))
            if len(input.shape) == 3:
                q_w = (input / alpha[:input.shape[0], :input.shape[1]].unsqueeze(2)).round().clamp(Qn, Qp)
            elif len(input.shape) == 2:
                q_w = (input / alpha[:input.shape[0]].unsqueeze(1)).round().clamp(Qn, Qp)
        elif layerwise == 'column':
            # print("183 input {} alpha {} ".format(input.shape, alpha.shape))
            if len(input.shape) == 3:
                q_w = (input / alpha[:input.shape[2]].unsqueeze(0).unsqueeze(0)).round().clamp(Qn, Qp)
            elif len(input.shape) == 2:
                q_w = (input / alpha[:input.shape[1]].unsqueeze(0)).round().clamp(Qn, Qp)
        if layerwise == 'layer':
            w_q = q_w * alpha
        elif layerwise == 'row':
            if len(input.shape) == 3:
                w_q = q_w * alpha[:input.shape[0], :input.shape[1]].unsqueeze(2)
            elif len(input.shape) == 2:
                w_q = q_w * alpha[:input.shape[0]].unsqueeze(1)
        elif layerwise == 'column':
            if len(input.shape) == 3:
                w_q = q_w * alpha[:input.shape[2]].unsqueeze(0).unsqueeze(0)
            elif len(input.shape) == 2:
                w_q = q_w * alpha[:input.shape[1]].unsqueeze(0)
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, learnable, layerwise = ctx.other

        if layerwise == 'layer':
            q_w = input_ / alpha
        elif layerwise == 'row':
            if len(input_.shape) == 3:
                q_w = input_ / alpha[:input_.shape[0], :input_.shape[1]].unsqueeze(2)
            elif len(input_.shape) == 2:
                q_w = input_ / alpha[:input_.shape[0]].unsqueeze(1)
        elif layerwise == 'column':
            if len(input_.shape) == 3:
                q_w = input_ / alpha[:input_.shape[2]].unsqueeze(0).unsqueeze(0)
            elif len(input_.shape) == 2:
                q_w = input_ / alpha[:input_.shape[1]].unsqueeze(0)

        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)

        if layerwise == 'layer':
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        elif layerwise == 'row':
            if len(input_.shape) == 3:
                grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                        -q_w + q_w.round())) * grad_output * grad_scale).sum(dim=2).unsqueeze(dim=0)
            elif len(input_.shape) == 2:
                grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                        -q_w + q_w.round())) * grad_output * grad_scale).sum(dim=1).unsqueeze(dim=0)
        elif layerwise == 'column':
            if len(input_.shape) == 3:
                grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                        -q_w + q_w.round())) * grad_output * grad_scale).sum(dim=(0, 1)).unsqueeze(dim=0)
            elif len(input_.shape) == 2:
                grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                        -q_w + q_w.round())) * grad_output * grad_scale).sum(dim=0).unsqueeze(dim=0)


        grad_input = indicate_middle * grad_output

        grad_alpha_pad = torch.zeros_like(alpha)
        if layerwise == 'layer':
            grad_alpha_pad = grad_alpha
        elif layerwise == 'row':
            input_shape = input_.shape
            if len(input_shape) == 3:
                grad_alpha_pad[:grad_alpha.shape[1], :grad_alpha.shape[2]] = grad_alpha
            if len(input_shape) == 2:
                grad_alpha_pad[:grad_alpha.shape[1]] = grad_alpha
        elif layerwise == 'column':
            input_shape = input_.shape
            if len(input_shape) == 3:
                grad_alpha_pad[:grad_alpha.shape[1]] = grad_alpha
            if len(input_shape) == 2:
                grad_alpha_pad[:grad_alpha.shape[1]] = grad_alpha

        if learnable:
            return grad_input, grad_alpha_pad, None, None, None
        else:
            return grad_input, None, None, None, None


class AsymLsqQuantizer(torch.autograd.Function):
    """
        Asymetric LSQ quantization. Modified from LSQ.
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise, learnable):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = 0
        Qp = 2 ** (num_bits) - 1
        # asymmetric: make sure input \in [0, +\inf], remember to add it back
        min_val = input.min().item()
        input_ = input - min_val
        # print(alpha.min())
        assert alpha.min() > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)
        if alpha.view(-1)[0] == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=False, init_method='default', layerwise=layerwise)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        # grad_scale = 1.0
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp, learnable, layerwise

        if layerwise == 'layer':
            q_w = (input_ / alpha).round().clamp(Qn, Qp)
        elif layerwise == 'row':
            if len(input.shape) == 3:
                q_w = (input_ / alpha[:input_.shape[0], :input.shape[1]].unsqueeze(2)).round().clamp(Qn, Qp)
            elif len(input.shape) == 2:
                q_w = (input_ / alpha[:input_.shape[0]].unsqueeze(1)).round().clamp(Qn, Qp)
        elif layerwise == 'column':
            if len(input.shape) == 3:
                q_w = (input_ / alpha[:input.shape[2]].unsqueeze(0).unsqueeze(0)).round().clamp(Qn, Qp)
            elif len(input.shape) == 2:
                q_w = (input_ / alpha[:input_.shape[1]].unsqueeze(0)).round().clamp(Qn, Qp)

        if layerwise == 'layer':
            w_q = q_w * alpha
        elif layerwise == 'row':
            if len(input.shape) == 3:
                w_q = q_w * alpha[:input_.shape[0], :input.shape[1]].unsqueeze(2)
            elif len(input.shape) == 2:
                w_q = q_w * alpha[:input_.shape[0]].unsqueeze(1)
        elif layerwise == 'column':
            if len(input.shape) == 3:
                w_q = q_w * alpha[:input_.shape[2]].unsqueeze(0).unsqueeze(0)
            elif len(input.shape) == 2:
                w_q = q_w * alpha[:input_.shape[1]].unsqueeze(0)
        w_q = w_q + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, learnable, layerwise = ctx.other

        if layerwise == 'layer':
            q_w = input_ / alpha
        elif layerwise == 'row':
            if len(input_.shape) == 3:
                q_w = input_ / alpha[:input_.shape[0], :input_.shape[1]].unsqueeze(2)
            elif len(input_.shape) == 2:
                q_w = input_ / alpha[:input_.shape[0]].unsqueeze(1)
        elif layerwise == 'column':
            if len(input_.shape) == 3:
                q_w = input_ / alpha[:input_.shape[2]].unsqueeze(0).unsqueeze(0)
            elif len(input_.shape) == 2:
                q_w = input_ / alpha[:input_.shape[1]].unsqueeze(0)


        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big   # this is more cpu-friendly than torch.ones(input_.shape)

        if layerwise == 'layer':
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        elif layerwise == 'row':
            if len(input_.shape) == 3:
                grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                        -q_w + q_w.round())) * grad_output * grad_scale).sum(dim=2).unsqueeze(dim=0)
            elif len(input_.shape) == 2:
                grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                        -q_w + q_w.round())) * grad_output * grad_scale).sum(dim=2).unsqueeze(dim=0)
        elif layerwise == 'column':
            if len(input_.shape) == 3:
                grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                        -q_w + q_w.round())) * grad_output * grad_scale).sum(dim=(0, 1)).unsqueeze(dim=0)
            elif len(input_.shape) == 2:
                grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                        -q_w + q_w.round())) * grad_output * grad_scale).sum(dim=0).unsqueeze(dim=0)


        grad_input = indicate_middle * grad_output

        grad_alpha_pad = torch.zeros_like(alpha)
        if layerwise == 'layer':
            grad_alpha_pad = grad_alpha
        elif layerwise == 'row':
            input_shape = input_.shape
            if len(input_shape) == 3:
                grad_alpha_pad[:grad_alpha.shape[1], :grad_alpha.shape[2]] = grad_alpha
            if len(input_shape) == 2:
                grad_alpha_pad[:grad_alpha.shape[1]] = grad_alpha
        elif layerwise == 'column':
            input_shape = input_.shape
            if len(input_shape) == 3:
                grad_alpha_pad[:grad_alpha.shape[1]] = grad_alpha
            if len(input_shape) == 2:
                grad_alpha_pad[:grad_alpha.shape[1]] = grad_alpha

        if learnable:
            return grad_input, grad_alpha_pad, None, None, None
        else:
            return grad_input, None, None, None, None
class LsqStepSize(nn.Parameter):
    def __init__(self, tensor):
        super(LsqStepSize, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        # print('Stepsize initialized to %.6f' % self.data.item())
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default', layerwise=True):
        # input: everthing needed to initialize step_size
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if layerwise == 'layer':
            if init_method == 'default':
                init_val = 2 * tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                    else 4 * tensor.abs().mean() / math.sqrt(Qp)
            elif init_method == 'uniform':
                init_val = 1./(2*Qp+1) if symmetric else 1./Qp
        elif layerwise == 'row':
            if len(tensor.shape) == 3:
                if init_method == 'default':
                    init_val = 2 * tensor.abs().mean(dim=2) / math.sqrt(Qp) if symmetric \
                        else 4 * tensor.abs().mean(dim=2) / math.sqrt(Qp)
                elif init_method == 'uniform':
                    init_val = 1./(2*Qp+1) if symmetric else 1./Qp
            if len(tensor.shape) == 2:
                if init_method == 'default':
                    init_val = 2 * tensor.abs().mean(dim=1) / math.sqrt(Qp) if symmetric \
                        else 4 * tensor.abs().mean(dim=1) / math.sqrt(Qp)
                elif init_method == 'uniform':
                    init_val = 1./(2*Qp+1) if symmetric else 1./Qp
        elif layerwise == 'column':
            if len(tensor.shape) == 3:
                if init_method == 'default':
                    init_val = 2 * tensor.abs().mean(dim=(0, 1)) / math.sqrt(Qp) if symmetric \
                        else 4 * tensor.abs().mean(dim=(0, 1)) / math.sqrt(Qp)
                elif init_method == 'uniform':
                    init_val = 1./(2*Qp+1) if symmetric else 1./Qp
            if len(tensor.shape) == 2:
                if init_method == 'default':
                    init_val = 2 * tensor.abs().mean(dim=0) / math.sqrt(Qp) if symmetric \
                        else 4 * tensor.abs().mean(dim=0) / math.sqrt(Qp)
                elif init_method == 'uniform':
                    init_val = 1./(2*Qp+1) if symmetric else 1./Qp

        eps = 1e-10 * torch.ones_like(init_val)
        init_val += eps
        # print("361 tensor {} init val {} self.data {} ".format(tensor.shape, init_val.shape, self.data.shape))
        self._initialize(init_val)


def act_quant_fn(input, clip_val, num_bits, symmetric, quant_method, layerwise, learnable):
    if num_bits == 32:
        return input

    elif quant_method=="uniform" and num_bits >= 2 and symmetric:
        quant_fn = SymQuantizer
    elif quant_method == "uniform" and num_bits >= 2 and not symmetric:
        quant_fn = AsymQuantizer
    elif quant_method == "lsq" and num_bits >= 2 and symmetric:
        quant_fn = SymLsqQuantizer
    elif quant_method == "lsq" and num_bits >= 2 and not symmetric:
        quant_fn = AsymLsqQuantizer
    else:
        raise ValueError("Unknownquant_method")

    input = quant_fn.apply(input, clip_val, num_bits, layerwise, learnable)

    return input


def weight_quant_fn(weight,  clip_val,  num_bits,  symmetric, quant_method, layerwise, learnable):
    if num_bits == 32:
        return weight

    elif num_bits >= 2 and symmetric and quant_method == "uniform":
        quant_fn = SymQuantizer
    elif quant_method == "uniform" and num_bits >= 2 and not symmetric:
        quant_fn = AsymQuantizer
    elif quant_method == "lsq" and num_bits >= 2 and symmetric:
        quant_fn = SymLsqQuantizer
    elif quant_method == "lsq" and num_bits >= 2 and not symmetric:
        quant_fn = AsymLsqQuantizer
    else:
        raise ValueError("Unknown quant_method")

    weight = quant_fn.apply(weight, clip_val,  num_bits, layerwise, learnable)
    return weight


