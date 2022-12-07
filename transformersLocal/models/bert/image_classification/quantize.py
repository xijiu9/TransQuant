from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
from torch.autograd.function import InplaceFunction, Function
import matplotlib.pyplot as plt
import time

try:
    from .preconditioner import ScalarPreconditioner, ScalarPreconditionerAct, lsq_per_tensor, lsq_plus, \
        TwoLayerWeightPreconditioner, LUQPreconditioner
    from .utils import twolayer_linearsample_weight, twolayer_linearsample_input, checkNAN
    from .activation_quantizer import SymQuantizer, AsymQuantizer, SymLsqQuantizer, AsymLsqQuantizer, LsqStepSize, \
        act_quant_fn, weight_quant_fn
except:
    from preconditioner import ScalarPreconditioner, ScalarPreconditionerAct, lsq_per_tensor, lsq_plus, \
        TwoLayerWeightPreconditioner, LUQPreconditioner
    from utils import twolayer_linearsample_weight, twolayer_linearsample_input, checkNAN
    from activation_quantizer import SymQuantizer, AsymQuantizer, SymLsqQuantizer, AsymLsqQuantizer, LsqStepSize, \
        act_quant_fn, weight_quant_fn
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

        self.weight_quant_method = 'LSQ'
        self.input_quant_method = ''
        self.learnable = True
        self.lsq_layerwise = True

        self.change_type = None
        self.change_threshold = 0

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

        self.first_pass = False
        self._build_weight_clip_val(qconfig.weight_quant_method, init_val=qconfig.clip_value)
        self._build_input_clip_val(qconfig.input_quant_method, init_val=qconfig.clip_value)

        self.is_second = False
        self.epsilon = None

    def _build_weight_clip_val(self, quant_method, init_val):
        if quant_method == 'uniform':
            # init_val = self.weight.mean().item() + 3 * self.weight.std().item()
            self.register_buffer('weight_clip_val', torch.tensor([-init_val, init_val]))
            self.weight_clip_val = nn.Parameter(self.weight_clip_val)
        elif quant_method == 'lsq' or qconfig.change_type:
            # TODO: for now we abuse the name for consistent reference in learner.
            # assert learnable, 'LSQ must use leranable step size!'
            self.weight_clip_val = LsqStepSize(
                torch.tensor(1.0, requires_grad=qconfig.learnable))  # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('weight_clip_val', None)

    def _build_input_clip_val(self, quant_method, init_val):
        if quant_method == 'uniform':
            self.register_buffer('input_clip_val', torch.tensor([-init_val, init_val]))
            self.input_clip_val = nn.Parameter(self.input_clip_val)
        elif quant_method == 'lsq' or qconfig.change_type:
            # TODO: for now we abuse the name for consistent reference in learner.
            # assert learnable, 'LSQ must use learnable step size!'
            self.input_clip_val = LsqStepSize(
                torch.tensor(1.0, requires_grad=qconfig.learnable))  # stepsize will be initialized in the first quantization

        else:
            self.register_buffer('input_clip_val', None)

    def set_first_forward(self):
        self.is_second = False

    def set_second_forward(self):
        self.is_second = True

    def forward(self, input):
        if not self.first_pass:
            print("Actually Using QLinear!")
            self.first_pass = True

        if qconfig.cutood != 0:
            input_sort = torch.sort(input.flatten())[0]
            max_thres, min_thres = input_sort[-len(input_sort) // qconfig.cutood], input_sort[len(input_sort) // qconfig.cutood]
            input = torch.clamp(input, min_thres, max_thres)
            # input[input > max_thres] = max_thres
            # input[input < min_thres] = min_thres

        if qconfig.weight_quant_method == 'ptq':
            qweight = quantize(self.weight, qconfig.weight_preconditioner())
        else:
            qweight = weight_quant_fn(self.weight, self.weight_clip_val, num_bits=qconfig.weight_num_bits,
                                      symmetric=True,
                                      quant_method=qconfig.weight_quant_method, layerwise=True, learnable=qconfig.learnable)
        # quantize input
        self.x = qweight
        if self.x.requires_grad:
            self.x.retain_grad()

        if self.is_second:
            qweight = qweight + self.epsilon

        if qconfig.input_quant_method == 'ptq':
            qinput = self.quantize_input(input)
        else:
            qinput = act_quant_fn(input, self.input_clip_val, num_bits=qconfig.activation_num_bits,
                                  symmetric=(self.name != 'addNorm_nsy'),
                                  quant_method=qconfig.input_quant_method, layerwise=True, learnable=qconfig.learnable)

        qbias = self.bias

        def draw(x, qx, s=''):
            plt.figure()
            num_bins = 256
            n, bins, patches = plt.hist(x, num_bins, density=1, color='green')
            # qn, _, _ = plt.hist(qx, num_bins, density=1 / 16, color='red')
            for qxi in np.unique(qx):
                plt.axvline(x=qxi, color='red', linestyle='--')
            plt.xlabel('X-Axis')
            plt.ylabel('Y-Axis')
            plt.xlim([x.min() * 1.1, x.max() * 1.1])
            plt.ylim([0, n.max() * 1.1])
            plt.title("ratio={}\nmin={}\nmax={}".format(np.abs((x.min() / x.max())), x.min(), x.max()),
                      fontweight="bold")
            time_tuple = time.localtime(time.time())

            os.makedirs("plt/{}/{}".format(self.name, s), exist_ok=True)
            plt.savefig('plt/{}/{}/{}:{}:{}.png'.format(self.name, s, time_tuple[3], time_tuple[4], time_tuple[5]))
            plt.close()

            f = open('plt/{}/{}/{}:{}:{}.txt'.format(self.name, s, time_tuple[3], time_tuple[4], time_tuple[5]), 'a')
            # f.write('{}\n{}\n{}\n{}\n'.format(x, qx, n, qn))
            f.write('{}\n{}\n{}\n{}\n'.format(x, qx, n, np.unique(qx)))
            f.close()

        # with torch.no_grad():
        #     if torch.rand(1) < 0.001:
        #         draw(input.view(-1).detach().cpu().numpy(), qinput.view(-1).detach().cpu().numpy(), 'input')
        #         draw(self.weight.view(-1).detach().cpu().numpy(), qweight.view(-1).detach().cpu().numpy(), 'weight')
        #         print("saved!")

        if hasattr(self, 'exact') or not qconfig.quantize_gradient:
            output = F.linear(qinput, qweight, qbias)
        else:
            output = linear_act.apply(qinput, qweight, qbias)
        # print(qinput.shape, qweight.shape, output.shape)
        return output


# Todo:暂定为QEmbedding之后的线性补充层
# 此处输入为embedding层的weight，需要按照weight的方式进行量化
class QIdentity(nn.Identity):
    def __init__(self, name=''):
        self.name = name
        super(QIdentity, self).__init__()

        self.quantize_input = QuantMeasure()
        self.first_pass = False
        self._build_embed_clip_val(qconfig.weight_quant_method, init_val=qconfig.clip_value)

    def _build_embed_clip_val(self, quant_method, init_val):
        # print(quant_method, init_val, qconfig.change_type)
        # print('!'*1000)
        if quant_method == 'uniform':
            self.register_buffer('embed_clip_val', torch.tensor([-init_val, init_val]))
            self.embed_clip_val = nn.Parameter(self.embed_clip_val)
        elif quant_method == 'lsq' or qconfig.change_type:
            # TODO: for now we abuse the name for consistent reference in learner.
            # assert learnable, 'LSQ must use learnable step size!'
            self.embed_clip_val = LsqStepSize(
                torch.tensor(1.0, requires_grad=qconfig.learnable))  # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('embed_clip_val', None)

    def forward(self, input):
        if not self.first_pass:
            print("Actually Using QIdentity!")
            self.first_pass = True
            torch.set_printoptions(precision=10)
        # print(self.embed_clip_val)
        if qconfig.weight_quant_method == 'ptq':
            qinput = quantize(input, qconfig.weight_preconditioner())
        else:
            qinput = weight_quant_fn(input, self.embed_clip_val, num_bits=qconfig.weight_num_bits, symmetric=True,
                                     quant_method=qconfig.weight_quant_method, layerwise=True, learnable=qconfig.learnable)

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
                time_tuple = time.localtime(time.time())

                plt.savefig(os.path.join(load_path_1, 'plt', "{}_{}_{}.png".format(name, s, time_tuple[3],
                                                                                   time_tuple[4], time_tuple[5])))
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
                print('*' * 20)
            if weight is not None:
                draw(weight.cpu().detach().numpy().flatten(), "weight")
