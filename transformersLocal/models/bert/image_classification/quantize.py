from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
from torch.autograd.function import InplaceFunction, Function

try:
    from .preconditioner import ScalarPreconditioner, ScalarPreconditionerAct, lsq_per_tensor, lsq_plus, \
        TwoLayerWeightPreconditioner, LUQPreconditioner
    from .utils import twolayer_linearsample_weight, twolayer_linearsample_input, checkNAN
except:
    from preconditioner import ScalarPreconditioner, ScalarPreconditionerAct, lsq_per_tensor, lsq_plus, \
        TwoLayerWeightPreconditioner, LUQPreconditioner
    from utils import twolayer_linearsample_weight, twolayer_linearsample_input, checkNAN
import IPython
import os
import matplotlib.pyplot as plt


class QuantizationConfig:
    def __init__(self):
        self.quantize_activation = True
        self.quantize_weights = True
        self.quantize_gradient = True
        self.activation_num_bits = 8
        self.weight_num_bits = 8
        self.bias_num_bits = 16
        self.backward_num_bits = 8
        self.bweight_num_bits = 8
        self.backward_persample = False
        self.biased = False
        self.grads = None
        self.acts = None
        self.hadamard = False
        self.biprecision = True
        self.twolayers_gradweight = False
        self.twolayers_gradinputt = False
        self.luq = False
        self.forward_method = 'PTQ'
        self.cutood = None
        self.clip_value = 0
        self.choice = None

    def activation_preconditioner(self):
        # return lambda x: ForwardPreconditioner(x, self.activation_num_bits)
        return lambda x: ScalarPreconditionerAct(x, self.activation_num_bits)
        # return lambda x: ScalarPreconditioner(x, 16)

    def weight_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.weight_num_bits)
        # return lambda x: ForwardPreconditioner(x, self.weight_num_bits)
        # return lambda x: DiagonalPreconditioner(x, self.weight_num_bits)

    def bias_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.bias_num_bits)

    def activation_gradient_preconditioner(self, special=False):
        if self.luq:
            return lambda x: LUQPreconditioner(x, self.backward_num_bits)
        if self.twolayers_gradinputt and not special:
            return lambda x: TwoLayerWeightPreconditioner(x, self.backward_num_bits)
        else:
            return lambda x: ScalarPreconditioner(x, self.backward_num_bits)

    def weight_gradient_preconditioner(self, special=False):
        if self.luq:
            return lambda x: LUQPreconditioner(x, self.bweight_num_bits)
        if self.twolayers_gradweight and not special:
            return lambda x: TwoLayerWeightPreconditioner(x, self.bweight_num_bits)
        return lambda x: ScalarPreconditioner(x, self.bweight_num_bits)


qconfig = QuantizationConfig()

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, Preconditioner, stochastic=False, inplace=False):
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print('---')
        #     print(input.view(-1)[:10], input.min(), input.max())
        with torch.no_grad():
            preconditioner = Preconditioner(output)
            output = preconditioner.forward()

            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
                # print("quantize 2", output)
                if qconfig.luq:
                    log_bias = math.log(4 / 3) - 1 / 2
                    output.add_(torch.ones_like(output) * log_bias)
                    # print("quantize 3", output)
            # quantize
            output.clamp_(0.0, preconditioner.num_bins).round_()

            output = preconditioner.inverse(output)

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print(output.view(-1)[:10])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None


def quantize(x, Preconditioner, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, Preconditioner, stochastic, inplace)


class linear_act(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.saved = input, weight, bias
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # print("grad output", grad_output)
        checkNAN(grad_output, "grad output")
        # torch.set_printoptions(profile="full", linewidth=160)
        # print("grad output", grad_output.shape)
        input, weight, bias = ctx.saved
        grad_output_weight_conditioner = quantize(grad_output,
                                                  qconfig.weight_gradient_preconditioner(),
                                                  stochastic=True)

        special_flag = (weight.shape[0] < 5)

        if special_flag:
            grad_output_active_conditioner = quantize(grad_output,
                                                      qconfig.activation_gradient_preconditioner(special=special_flag),
                                                      stochastic=True)
        else:
            # grad_output_active_conditioner = quantize(grad_output.transpose(-2, -1), qconfig.activation_gradient_preconditioner(),
            #                                           stochastic=True).transpose(-2, -1)
            grad_output_active_conditioner = quantize(grad_output, qconfig.activation_gradient_preconditioner(),
                                                      stochastic=True)

        # print("shape of input is:", input.size())
        # print("shape of grad_output is:", grad_output.size())
        C_in = input.shape[-1]
        C_out = grad_output.shape[-1]

        grad_output_flatten = grad_output.reshape(-1, C_out)
        grad_output_flatten_weight = grad_output_weight_conditioner.reshape(-1, C_out)
        # if qconfig.twolayers_gradinputt and not special_flag:
        #     grad_output_flatten_active = grad_output_active_conditioner.reshape(-1, 2 * C_out)
        # else:
        #     grad_output_flatten_active = grad_output_active_conditioner.reshape(-1, C_out)
        grad_output_flatten_active = grad_output_active_conditioner.reshape(-1, C_out)

        # print(grad_output_flatten_active.shape, grad_output_flatten_weight.shape, input.shape, special_flag)
        input_flatten = input.reshape(-1, C_in)

        if qconfig.twolayers_gradweight:
            if qconfig.grads is not None:
                qconfig.grads[0].append(grad_output_flatten_weight)
                qconfig.grads[1].append(input_flatten)
                print("save grad")

            m1, m2 = twolayer_linearsample_weight(grad_output_flatten_weight, input_flatten)
            grad_weight = m1.t().mm(m2)
        else:
            # print("weight", grad_output_flatten_weight.shape, input_flatten.shape)
            grad_weight = grad_output_flatten_weight.t().mm(input_flatten)

        if qconfig.twolayers_gradinputt:

            if special_flag:
                grad_input = grad_output_flatten_active.mm(weight)
            else:
                I = torch.eye(grad_output_flatten_active.shape[0] // 2, device="cuda")
                grad_input, _ = twolayer_linearsample_input(grad_output_flatten_active, I)

                checkNAN(grad_input, "grad input before")
                grad_input = grad_input.mm(weight)
                checkNAN(grad_input, "grad input after")
                # print(grad_output_flatten_active.shape)
                if qconfig.grads is not None:
                    qconfig.grads[2].append(grad_output_flatten_active)
                    qconfig.grads[3].append(I)
        else:
            # print("input", grad_output_flatten_active.shape, weight.shape)
            grad_input = grad_output_flatten_active.mm(weight)
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None
        grad_input_transform = grad_input.reshape(input.size())
        # print("shape of grad_input is:", grad_input.size())
        # print("shape of grad_weight is:", grad_weight.size())
        # print("shape of grad_bias is:", grad_bias.size())
        checkNAN(grad_input_transform, "grad input transform")
        # print("grad_input_transform", grad_input_transform)
        return grad_input_transform, grad_weight, grad_bias


class identity_act(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.saved = input
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # torch.set_printoptions(profile="full", linewidth=160)
        grad_output_weight_conditioner = quantize(grad_output,
                                                  qconfig.weight_gradient_preconditioner(special=True),
                                                  stochastic=True)
        input = ctx.saved
        # print("shape of input is:", input.size())
        # print("shape of grad_output is:", grad_output.size())
        # C_in = input.shape[-1]
        C_out = grad_output.shape[-1]

        grad_output_flatten_weight = grad_output_weight_conditioner.reshape(-1, C_out)

        grad_input = grad_output_flatten_weight
        grad_input_transform = grad_input.reshape(input.size())
        # print("shape of grad_input is:", grad_input.size())
        # print("shape of grad_weight is:", grad_weight.size())
        # print("shape of grad_bias is:", grad_bias.size())
        return grad_input_transform


class LSQPerTensor(nn.Module):
    def __init__(self, bits, aw='', name=''):
        super(LSQPerTensor, self).__init__()

        self.aw = aw
        self.bits = bits
        self.step_size = Parameter(torch.tensor(1.0), requires_grad=True)
        self.initialized = False
        self.name = name

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                num_bins = 2 ** self.bits - 1
                if self.name in ["attention", "addNorm", "pooler", "classifier"]:
                    step_size = 2 * x.abs().mean() / np.sqrt(num_bins)
                elif self.name in ["embedding"]:
                    step_size = 4 * x.abs().mean() / np.sqrt(num_bins)  # embedding
                elif self.name == "feedForward":
                    # step_size = x.abs().max() / num_bins  # embedding
                    step_size = 6 * x.abs().mean() / np.sqrt(num_bins)  # embedding
                else:
                    if name != '':
                        print("?{}".format(name))
                self.step_size.copy_(step_size)  # LSQ type

                self.initialized = True

        if x.max() > -1e-8 and x.min() < 1e-8:
            symm = True
        elif x.max() > -1e-8 and x.min() > -1e-8:
            symm = False
        else:
            print("min max not compatible for SAWB")
            symm = None
        rand = torch.rand(1)
        # if rand < 0.001:
        #     print(self.name, self.aw, (x.min() / x.max()).abs(), x.min(), x.max())
        return lsq_per_tensor().apply(x, self.step_size, self.bits, symm, rand)


class LSQplus(nn.Module):
    def __init__(self, bits, aw='', name=''):
        super(LSQplus, self).__init__()

        self.aw = aw
        self.bits = bits
        if aw == 'a':
            self.step_size = Parameter(torch.tensor(1.0), requires_grad=True)
            self.beta = Parameter(torch.tensor(1.0), requires_grad=True)
        elif aw == 'w':
            self.step_size = Parameter(torch.tensor(1.0), requires_grad=True)
            self.beta = Parameter(torch.tensor(0.0), requires_grad=False)
        self.initialized = False
        self.name = name

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                num_bins = 2 ** self.bits - 1
                if self.aw == 'a':
                    if self.name in ["attention", "addNorm", "feedForward", "pooler", "classifier"]:
                        step_size = 2 * x.abs().mean() / np.sqrt(num_bins)
                    elif self.name == "embedding":
                        step_size = 4 * x.abs().mean() / np.sqrt(num_bins)  # embedding
                    else:
                        if name != '':
                            print("?{}".format(name))
                    self.step_size.copy_(step_size)  # LSQ type

                    self.initialized = True
                elif self.aw == 'w':
                    mean = torch.mean(x.detach())
                    std = torch.std(x.detach())
                    self.step_size.data = max([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]) / num_bins

                    self.initialized = True

                    del mean, std, num_bins

        if x.max() > -1e-8 and x.min() < 1e-8:
            symm = True
        elif x.max() > -1e-8 and x.min() > -1e-8:
            symm = False
        else:
            print("min max not compatible for SAWB")
            symm = None
        rand = torch.rand(1)
        # if rand < 0.001:
        #     print(self.name, self.aw, (x.min() / x.max()).abs(), x.min(), x.max())
        return lsq_plus().apply(x, self.step_size, self.beta, self.bits, symm, self.aw, rand)

    def quantize_MSE(self, input, scale, bits, symm):
        num_bins = 2 ** bits - 1
        bias = -num_bins / 2 if symm else 0

        # Forward
        eps = 1e-7
        scale = scale + eps
        transformed = input / scale - bias
        vbar = torch.clamp(transformed, 0.0, num_bins).round()
        quantized = (vbar + bias) * scale

        MSE = (quantized - input).square().sum()
        return MSE


class UniformQuantizeSawb(InplaceFunction):

    @staticmethod
    def forward(ctx, input, c1, c2, Qp, Qn):
        output = input.clone()

        with torch.no_grad():
            clip = (c1 * torch.sqrt(torch.mean(input ** 2))) + (c2 * torch.mean(input.abs()))
            scale = 2 * clip / (Qp - Qn)
            output.div_(scale)
            output.clamp_(Qn, Qp).round_()
            output.mul_(scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None


def get_sawb_coefficients(bits):
    bits = int(bits)
    coefficient_dict = {1: [0., 1.], 2: [3.19, -2.14], 3: [7.40, -6.66], 4: [11.86, -11.68],
                        5: [17.08, -17.66], 6: [22.49, -23.95], 7: [28.68, -31.24],
                        8: [32.27, -35.46], 16: [34.26, -37.60], 32: [40.60, -45.33]}
    return coefficient_dict[bits]


class SAWBTensor(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, bits=8, aw='', name=''):
        super(SAWBTensor, self).__init__()

        self.aw = aw
        self.bits = bits
        self.c1, self.c2 = get_sawb_coefficients(self.bits)
        self.name = name

    def forward(self, input):
        if input.max() > -1e-8 and input.min() < 1e-8:
            Qn = -2 ** (self.bits - 1)
            Qp = 2 ** (self.bits - 1)
        elif input.max() > -1e-8 and input.min() > -1e-8:
            Qn = 0
            Qp = 2 ** self.bits - 1
        else:
            print("min max not compatible for SAWB")
            Qn = 0
            Qp = 0

        return UniformQuantizeSawb().apply(input, self.c1, self.c2, Qp, Qn)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, inplace=False, stochastic=False):
        super(QuantMeasure, self).__init__()
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input):
        q_input = quantize(input, qconfig.activation_preconditioner(),
                           stochastic=self.stochastic, inplace=self.inplace)
        return q_input


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, name=''):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.quantize_input = QuantMeasure()
        self.name = name
        if name in qconfig.choice or qconfig.choice[0] in ['quantize', 'linear']:
            if qconfig.forward_method == 'LSQ':
                self.lsqweight = LSQPerTensor(qconfig.weight_num_bits, aw='w', name=name)
                self.lsqactive = LSQPerTensor(qconfig.activation_num_bits, aw='a', name=name)
            elif qconfig.forward_method == 'LSQplus':
                self.lsqplusweight = LSQplus(qconfig.weight_num_bits, aw='w', name=name)
                self.lsqplusactive = LSQplus(qconfig.activation_num_bits, aw='a', name=name)
            elif qconfig.forward_method == 'SAWB':
                self.SAWBweight = SAWBTensor(qconfig.weight_num_bits, aw='w', name=name)
                self.SAWBactive = SAWBTensor(qconfig.activation_num_bits, aw='a', name=name)
        self.first_pass = False

    def forward(self, input):
        if not self.first_pass:
            print("Actually Using QLinear!")
            self.first_pass = True
        if qconfig.clip_value != 0:
            input = input.clamp_(-qconfig.clip_value, qconfig.clip_value)

        if qconfig.cutood != 0:
            input_sort = torch.sort(input.flatten())[0]
            max_thres, min_thres = input_sort[-len(input_sort) // qconfig.cutood], input_sort[len(input_sort) // qconfig.cutood]
            input[input > max_thres] = max_thres
            input[input < min_thres] = min_thres
            path = os.path.join("/workspace/home/xihaocheng20/TransQuant/transformersLocal/models/bert/image_classification/ckpt", qconfig.forward_method, str(qconfig.cutood), self.name)
            os.makedirs(path, exist_ok=True)
            num_files = len(os.listdir(path))
            if num_files < 10:
                torch.save([input, self.weight], os.path.join(path, '{}_{}.pt'.format(self.name, num_files)))
                print(self.name, (input.max() / input.min()).abs(), input.max(), input.min())

        if qconfig.quantize_activation:
            if qconfig.forward_method == 'LSQ':
                qinput = self.lsqactive(input)
            elif qconfig.forward_method == 'LSQplus':
                qinput = self.lsqplusactive(input)
            elif qconfig.forward_method == 'SAWB':
                qinput = self.SAWBactive(input)
            else:
                qinput = self.quantize_input(input)
        else:
            qinput = input

        if qconfig.quantize_weights:
            if qconfig.forward_method == 'LSQ':
                qweight = self.lsqweight(self.weight)
            elif qconfig.forward_method == 'LSQplus':
                qweight = self.lsqplusweight(self.weight)
            elif qconfig.forward_method == 'SAWB':
                qweight = self.SAWBweight(self.weight)
            else:
                qweight = quantize(self.weight, qconfig.weight_preconditioner())

            if self.bias is not None:
                qbias = quantize(self.bias, qconfig.bias_preconditioner())
            else:
                qbias = None
        else:
            qweight = self.weight
            qbias = self.bias

        if hasattr(self, 'exact') or not qconfig.quantize_gradient:
            output = F.linear(qinput, qweight, qbias)
        else:
            output = linear_act.apply(qinput, qweight, qbias)

        return output


# Todo:暂定为QEmbedding之后的线性补充层
# 此处输入为embedding层的weight，需要按照weight的方式进行量化
class QIdentity(nn.Identity):
    def __init__(self, name=''):
        self.name = name
        super(QIdentity, self).__init__()
        print(qconfig.choice)
        if name in qconfig.choice or qconfig.choice[0] in ['quantize']:
            if qconfig.forward_method == 'LSQ':
                self.lsqweight = LSQPerTensor(qconfig.weight_num_bits, aw='w', name=name)
            elif qconfig.forward_method == 'LSQplus':
                self.lsqplusweight = LSQplus(qconfig.weight_num_bits, aw='w', name=name)
            elif qconfig.forward_method == 'SAWB':
                self.SAWBweight = SAWBTensor(qconfig.weight_num_bits, aw='w', name=name)
        self.first_pass = False
        
    def forward(self, input):
        if not self.first_pass:
            print("Actually Using QIdentity!")
            self.first_pass = True
        if qconfig.clip_value != 0:
            input = input.clamp_(-qconfig.clip_value, qconfig.clip_value)

        if qconfig.cutood != 0:
            input_sort = torch.sort(input.flatten())[0]
            max_thres, min_thres = input_sort[-len(input_sort) // qconfig.cutood], input_sort[len(input_sort) // qconfig.cutood]
            input[input > max_thres] = max_thres
            input[input < min_thres] = min_thres
            path = os.path.join("/workspace/home/xihaocheng20/TransQuant/transformersLocal/models/bert/image_classification/ckpt", qconfig.forward_method, str(qconfig.cutood), self.name)
            os.makedirs(path, exist_ok=True)
            num_files = len(os.listdir(path))
            if num_files < 10:
                torch.save([None, input], os.path.join(path, '{}_{}.pt'.format(self.name, num_files)))
                print(self.name, (input.max() / input.min()).abs(), input.max(), input.min())


        if qconfig.quantize_weights:
            if qconfig.forward_method == 'LSQ':
                qinput = self.lsqweight(input)
            elif qconfig.forward_method == 'LSQplus':
                qinput = self.lsqplusweight(input)
            elif qconfig.forward_method == 'SAWB':
                qinput = self.SAWBweight(input)
            else:
                qinput = quantize(input, qconfig.weight_preconditioner())
        else:
            qinput = input

        if hasattr(self, 'exact') or not qconfig.quantize_gradient:
            output = qinput
        else:
            output = identity_act.apply(qinput)

        return output


if __name__ == '__main__':
    load_path = os.path.join("./ckpt/PTQ", str(500))
    for name in ["embedding", "attention", "addNorm", "feedForward", "pooler", "classifier"]:
        load_path_1 = os.path.join(load_path, name)
        os.makedirs(os.path.join(load_path_1, 'plt'), exist_ok=True)
        for idx, file in enumerate(os.listdir(load_path_1)):
            file_path = os.path.join(load_path_1, file)
            if os.path.isdir(file_path):
                continue
            input, weight = torch.load(file_path)

            def draw(x, s=''):
                plt.figure()
                num_bins = 100
                n, bins, patches = plt.hist(x, num_bins, density=1, color='green')
                plt.xlabel('X-Axis')
                plt.ylabel('Y-Axis')
                plt.xlim([x.min() * 1.1, x.max() * 1.1])
                plt.ylim([0, n.max() * 1.1])
                plt.title("ratio={}\nmin={}\nmax={}".format(np.abs((x.min() / x.max())), x.min(), x.max()),
                          fontweight="bold")
                plt.savefig(os.path.join(load_path_1, 'plt', "{}_{}_{}.png".format(name, s, idx)))
                plt.close()

            def mse(x, y):
                return np.mean(np.square(x - y))

            def find_mse(x, name):
                MSE_list = []

                def quantize(x, scale, bits):
                    num_bins = 2 ** bits
                    upper = 2 ** (bits - 1)
                    lower = -2 ** (bits - 1)

                    x = x / scale
                    x = np.clip(x, lower, upper)
                    x = np.around(x)
                    x = x * scale
                    return x

                lsqactive = LSQPerTensor(4, aw='a', name=name)
                SAWBactive = SAWBTensor(4, aw='a', name=name)

                for i in range(1, 40):
                    scale = i / 20
                    x_q = quantize(x, scale=scale, bits=4)
                    MSE_list.append(mse(x, x_q))

                min_MSE, min_idx = np.min(MSE_list), np.argmin(MSE_list)
                lsq_quant, SAWB_quant = lsqactive.forward(torch.tensor(x)), SAWBactive.forward(torch.tensor(x))
                lsq_MSE, SAWB_MSE = mse(x, lsq_quant.detach().numpy()), mse(x, SAWB_quant.detach().numpy())
                print("min mse:{}, lsq mse:{}, SAWB mse:{}".format(min_MSE, lsq_MSE, SAWB_MSE))

            if input is not None:
                input_np = input.cpu().detach().numpy().flatten()
                draw(input_np, "input")
                find_mse(input_np, name=name)
                input_np.sort()
                print(len(input_np), input.shape)

                input_np[input_np > input_np[-len(input_np) // 500]] = 0
                input_np[input_np < input_np[len(input_np) // 500]] = 0
                draw(input_np, "input_clip")
                find_mse(input_np, name=name)
                print('*'*20)
            if weight is not None:
                draw(weight.cpu().detach().numpy().flatten(), "weight")

