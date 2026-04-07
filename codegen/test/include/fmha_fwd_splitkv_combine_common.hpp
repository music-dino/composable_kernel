// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/host/device_fmha_splitkv_combine/problem.hpp"
#include "ck/host/types.hpp"
#include "ck/host/utils.hpp"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace ck {
namespace host {

inline const std::string combine_kernel_template = R"__ck__(
#include <cmath>
#include <cstdint>
#include <cassert>
#include <sstream>
#include <${include}>

using KernelType = ${template};

extern "C" __launch_bounds__(KernelType::Kernel::kBlockSize, KernelType::Kernel::kBlockPerCu)
__global__ void f(const float* lse_acc, const float* o_acc, ${dtype}* o) {
    
    using Kernel = KernelType;
    
    constexpr auto desc = Kernel::make_descriptor(
        ${batch}, ${nhead}, ${m}, ${o_dim}, ${num_splits},
        ck_tile::make_tuple(${lse_acc_stride_batch}, ${lse_acc_stride_nhead}, ${lse_acc_stride_split}),
        ck_tile::make_tuple(${o_acc_stride_batch}, ${o_acc_stride_nhead}, ${o_acc_stride_split}, ${o_acc_stride_m}),
        ck_tile::make_tuple(${o_stride_batch}, ${o_stride_nhead}, ${o_stride_m}));
    
    static_assert(desc.IsValid(), "Invalid Combine kernel configuration");
    
    Kernel::Run(desc, lse_acc, o_acc, o);
}
)__ck__";

struct SplitKVCombineParams
{
    std::size_t batch;
    std::size_t nhead;
    std::size_t M; // seqlen_q
    std::size_t O; // hdim_v
    std::size_t num_splits;

    // LSE_acc strides [batch, nhead, num_splits, M]
    std::size_t lse_acc_stride_split;
    std::size_t lse_acc_stride_nhead;
    std::size_t lse_acc_stride_batch;

    // O_acc strides [batch, nhead, num_splits, M, O]
    std::size_t o_acc_stride_m;
    std::size_t o_acc_stride_split;
    std::size_t o_acc_stride_nhead;
    std::size_t o_acc_stride_batch;

    // O strides [batch, nhead, M, O]
    std::size_t o_stride_m;
    std::size_t o_stride_nhead;
    std::size_t o_stride_batch;
};

inline SplitKVCombineParams
make_splitkv_combine_params(const ck::host::device_fmha_splitkv_combine::Problem& prob)
{
    SplitKVCombineParams p;
    p.batch      = prob.batch;
    p.nhead      = prob.nhead;
    p.M          = prob.M;
    p.O          = prob.O;
    p.num_splits = prob.num_splits;

    // LSE_acc - [batch, nhead, num_splits, M]
    p.lse_acc_stride_split = prob.M;
    p.lse_acc_stride_nhead = prob.num_splits * prob.M;
    p.lse_acc_stride_batch = prob.nhead * prob.num_splits * prob.M;

    // O_acc - [batch, nhead, num_splits, M, O]
    p.o_acc_stride_m     = prob.O;
    p.o_acc_stride_split = prob.M * prob.O;
    p.o_acc_stride_nhead = prob.num_splits * prob.M * prob.O;
    p.o_acc_stride_batch = prob.nhead * prob.num_splits * prob.M * prob.O;

    // O - [batch, nhead, M, O]
    p.o_stride_m     = prob.O;
    p.o_stride_nhead = prob.M * prob.O;
    p.o_stride_batch = prob.nhead * prob.M * prob.O;

    return p;
}

// Grid: (ceil(M/kM0) * ceil(O/kN1), nhead, batch)
// Block: (4 warps * 64 = 256)
inline std::pair<dim3, dim3>
get_splitkv_combine_launch_dims(const ck::host::Solution& solution,
                                const ck::host::device_fmha_splitkv_combine::Problem& prob)
{
    auto kN1 = solution.GetTemplateParameter<std::size_t>("N1");
    auto kM0 = kN1 / 4;

    constexpr std::size_t warp_size  = 64;
    constexpr std::size_t num_warps  = 4;
    constexpr std::size_t block_size = num_warps * warp_size;

    const auto grid_m = integer_divide_ceil(prob.M, kM0);
    const auto grid_n = integer_divide_ceil(prob.O, kN1);

    dim3 grid(grid_m * grid_n, prob.nhead, prob.batch);
    dim3 block(block_size, 1, 1);

    return {grid, block};
}

std::string
make_splitkv_combine_kernel_source(const ck::host::device_fmha_splitkv_combine::Problem& prob,
                                   const ck::host::Solution& solution,
                                   const SplitKVCombineParams& params)
{
    return ck::host::InterpolateString(
        ck::host::combine_kernel_template,
        {{"include", prob.GetIncludeHeader()},
         {"template", solution.ToTemplateString()},
         {"dtype", "ck_tile::fp16_t"},
         {"batch", std::to_string(params.batch)},
         {"nhead", std::to_string(params.nhead)},
         {"m", std::to_string(params.M)},
         {"o_dim", std::to_string(params.O)},
         {"num_splits", std::to_string(params.num_splits)},
         {"lse_acc_stride_batch", std::to_string(params.lse_acc_stride_batch)},
         {"lse_acc_stride_nhead", std::to_string(params.lse_acc_stride_nhead)},
         {"lse_acc_stride_split", std::to_string(params.lse_acc_stride_split)},
         {"o_acc_stride_batch", std::to_string(params.o_acc_stride_batch)},
         {"o_acc_stride_nhead", std::to_string(params.o_acc_stride_nhead)},
         {"o_acc_stride_split", std::to_string(params.o_acc_stride_split)},
         {"o_acc_stride_m", std::to_string(params.o_acc_stride_m)},
         {"o_stride_batch", std::to_string(params.o_stride_batch)},
         {"o_stride_nhead", std::to_string(params.o_stride_nhead)},
         {"o_stride_m", std::to_string(params.o_stride_m)}});
}

} // namespace host
} // namespace ck
