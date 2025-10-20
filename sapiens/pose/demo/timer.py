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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import dropwhile
from time import perf_counter_ns


class Timer:
    def __init__(self, key, d):
        self.key = key
        self.d = d
        if self.key not in self.d.keys():
            self.d[self.key] = []

    def __enter__(self):
        self.start = perf_counter_ns()
        return self

    def __exit__(self, *args):
        elapsed = perf_counter_ns() - self.start
        # base, unit = next(
        #     dropwhile(
        #         lambda t: elapsed > 10 ** (t[0] + 3),
        #         [(0, "ns"), (3, "us"), (6, "ms"), (9, "s")],
        #     ),
        #     (9, "s"),
        # )
        self.d[self.key].append(elapsed / 10**9)
