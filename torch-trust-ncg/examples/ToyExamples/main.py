# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling BLADE or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import torch
from torchtrustncg import TrustRegion
from torchtrustncg.utils import rosenbrock, branin

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, default='rosenbrock', choices=['rosenbrock', 'branin'],
                        help='Test function to optimize')
    parser.add_argument('--num-iters', dest='num_iters', default=200, type=str,
                        help='Number of iterations to use')
    parser.add_argument('--gtol', default=1e-4, type=float,
                        help='Gradient tolerance')
    parser.add_argument('--init-point', nargs=2, default=[2.5, 2.5], type=float, dest='init_point',
                        help='Point to initialize the optimization')

    parser.add_argument('--plot',
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Plot the function trajectory')

    args = parser.parse_args()
    function = args.function
    num_iters = args.num_iters
    gtol = args.gtol
    init_point = args.init_point
    plot = args.plot

    if function == 'rosenbrock':
        loss_func = rosenbrock
    elif function == 'branin':
        loss_func = branin
    else:
        print(f'Unknown loss function {function}')
        sys.exit(-1)

    variable = torch.empty([1, 2], requires_grad=True)
    with torch.no_grad():
        variable[0, 0] = init_point[0]
        variable[0, 1] = init_point[1]

    optimizer = TrustRegion([variable], opt_method='krylov')

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()
        loss = loss_func(variable)
        if backward:
            loss.backward(create_graph=True)
        return loss

    points = []
    values = []

    for n in range(num_iters):
        loss = optimizer.step(closure)

        np_var = variable.detach().cpu().numpy().squeeze()

        if plot:
            points.append(np_var.copy())
            values.append(loss.item())

        if torch.norm(variable.grad).item() < gtol:
            break
        if torch.norm(optimizer.param_step, dim=-1).lt(gtol).all():
            break

        print(
            f'[{n:04d}]: ' +
            f'Loss at ({variable[0, 0]:.4f}, {variable[0, 1]:.4f}) = ' +
            f'{loss.item():.4f}')

    if plot:
        N = 100
        X, Y = np.meshgrid(
            np.linspace(-4 + init_point[0], init_point[0] + 4, N),
            np.linspace(-4 + init_point[1], init_point[1] + 4, N)
        )

        grid_points = np.stack([X, Y], axis=-1).reshape(-1, 2)
        Z = loss_func(torch.from_numpy(grid_points)).detach().numpy()

        cs = plt.contourf(X, Y, Z.reshape(N, N))

        points = np.stack(points, axis=0)

        plt.plot(points[:, 0], points[:, 1], 'x-', color='black')
        plt.show()
