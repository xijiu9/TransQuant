import os

import matplotlib.pyplot as plt
import torch
import numpy as np

load_dir = "/workspace/home/xihaocheng20/TransQuant/test_glue_result_quantize/mrpc/twolayer/choice=quantize_/seed=27"

for epoch in [0, 1, 2]:
    x = torch.load(os.path.join(load_dir, '{}epoch.pt'.format(epoch)))
    FG, TG, FG_grad, TG_grad = x["FG"], x["TG"], x["FG_grad"], x["TG_grad"]

    grad_output_flatten_weight, input_flatten, grad_output_flatten_active, I = TG_grad
    
    # plt_x, plt_y = np.array([]), np.array([])
    # for idx, (fg, tg) in enumerate(zip(FG, TG)):
    #     plt_x = np.append(plt_x, idx)
    #     # print(fg, tg)
    #     MSE = (fg - tg).square().sum().cpu().numpy()
    #     print(MSE)
    #     plt_y = np.append(plt_y, MSE)
    #
    # plt.figure(1)
    # plt.plot(plt_x, plt_y, label='{}'.format(epoch))

plt.legend()
plt.savefig(os.path.join(load_dir, "variance.png"))
