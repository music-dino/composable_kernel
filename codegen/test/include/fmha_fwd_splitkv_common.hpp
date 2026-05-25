// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/host/device_fmha_splitkv/problem.hpp"
#include "ck/host/types.hpp"
#include "ck/host/utils.hpp"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace ck {
namespace host {

inline const std::string splitkv_kernel_template = R"__ck__(
#include <cmath>
#include <cstdint>
#include <cassert>
#include <sstream>
#include <${include}>

using KernelType = ${template};

extern "C" __launch_bounds__(KernelType::Kernel::kBlockSize, KernelType::Kernel::kBlockPerCu)
__global__ void f(const ${dtype}* q, const ${dtype}* k, const ${dtype}* v,
                  _Float16* o_acc, float* lse_acc) {
    
    constexpr float scale_s = ${scale_s};
    
    using Kernel = KernelType;
    
    constexpr auto desc = Kernel::make_descriptor(
        ck_tile::make_tuple(${batch}, ${nhead}, ${m}, ${k}),
        ck_tile::make_tuple(${q_stride_batch}, ${q_stride_nhead}, ${q_stride_m}),
        ck_tile::make_tuple(${batch}, ${nhead_k}, ${n}, ${k}),
        ck_tile::make_tuple(${k_stride_batch}, ${k_stride_nhead}, ${k_stride_n}),
        ck_tile::make_tuple(${batch}, ${nhead_k}, ${n}, ${o}),
        ck_tile::make_tuple(${v_stride_batch}, ${v_stride_nhead}, ${v_stride_n}),
        ck_tile::make_tuple(${batch}, ${nhead}, ${num_splits}, ${m}, ${o}),
        ck_tile::make_tuple(${o_acc_stride_batch}, ${o_acc_stride_nhead}, ${o_acc_stride_split}, ${o_acc_stride_m}),
        ck_tile::make_tuple(${lse_acc_stride_batch}, ${lse_acc_stride_nhead}, ${lse_acc_stride_split}));
    
    static_assert(desc.IsValid(), "Invalid SplitKV kernel configuration");
    
    Kernel::Run(desc, scale_s, q, k, v, lse_acc, o_acc);
}
)__ck__";

struct SplitKVParams
{
    std::size_t batch;
    std::size_t nhead;
    std::size_t nhead_k;
    std::size_t M; // seqlen_q
    std::size_t N; // seqlen_k
    std::size_t K; // hdim_q
    std::size_t O; // hdim_v
    std::size_t num_splits;
    float scale_s;

    // Q strides [batch, nhead, M, K]
    std::size_t q_stride_m;
    std::size_t q_stride_nhead;
    std::size_t q_stride_batch;

    // K strides [batch, nhead_k, N, K]
    std::size_t k_stride_n;
    std::size_t k_stride_nhead;
    std::size_t k_stride_batch;

    // V strides [batch, nhead_k, N, O]
    std::size_t v_stride_n;
    std::size_t v_stride_nhead;
    std::size_t v_stride_batch;

    // O_acc strides [batch, nhead, num_splits, M, O]
    std::size_t o_acc_stride_m;
    std::size_t o_acc_stride_split;
    std::size_t o_acc_stride_nhead;
    std::size_t o_acc_stride_batch;

    // LSE_acc strides [batch, nhead, num_splits, M]
    std::size_t lse_acc_stride_split;
    std::size_t lse_acc_stride_nhead;
    std::size_t lse_acc_stride_batch;
};

inline SplitKVParams make_splitkv_params(const device_fmha_splitkv::Problem& prob, float scale_s)
{
    SplitKVParams p;
    p.batch      = prob.batch;
    p.nhead      = prob.nhead;
    p.nhead_k    = prob.nhead_k;
    p.M          = prob.M;
    p.N          = prob.N;
    p.K          = prob.K;
    p.O          = prob.O;
    p.num_splits = prob.num_splits;
    p.scale_s    = scale_s;

    // Q - [batch, nhead, M, K]
    p.q_stride_m     = prob.K;
    p.q_stride_nhead = prob.M * prob.K;
    p.q_stride_batch = prob.nhead * prob.M * prob.K;

    // K - [batch, nhead_k, N, K]
    p.k_stride_n     = prob.K;
    p.k_stride_nhead = prob.N * prob.K;
    p.k_stride_batch = prob.nhead_k * prob.N * prob.K;

    // V - [batch, nhead_k, N, O]
    p.v_stride_n     = prob.O;
    p.v_stride_nhead = prob.N * prob.O;
    p.v_stride_batch = prob.nhead_k * prob.N * prob.O;

    // O_acc - [batch, nhead, num_splits, M, O]
    p.o_acc_stride_m     = prob.O;
    p.o_acc_stride_split = prob.M * prob.O;
    p.o_acc_stride_nhead = prob.num_splits * prob.M * prob.O;
    p.o_acc_stride_batch = prob.nhead * prob.num_splits * prob.M * prob.O;

    // LSE_acc - [batch, nhead, num_splits, M]
    p.lse_acc_stride_split = prob.M;
    p.lse_acc_stride_nhead = prob.num_splits * prob.M;
    p.lse_acc_stride_batch = prob.nhead * prob.num_splits * prob.M;

    return p;
}

// Grid: (ceil(M_eff/BM0) * ceil(O/BN1) * num_splits, nhead_eff, batch)
// When kMergeNumHeadGroupsSeqLenQ is true:
//   - nhead_eff = nhead_k
//   - M_eff = M * (nhead / nhead_k)
// Block: (num_warps * warp_size)
inline std::pair<dim3, dim3>
get_splitkv_launch_dims(const ck::host::Solution& solution,
                        const ck::host::device_fmha_splitkv::Problem& prob)
{
    auto bm0 = solution.GetTemplateParameter<std::size_t>("BM0");
    auto bn1 = solution.GetTemplateParameter<std::size_t>("BN1");
    auto rm0 = solution.GetTemplateParameter<std::size_t>("RM0");
    auto rn0 = solution.GetTemplateParameter<std::size_t>("RN0");
    auto rk0 = solution.GetTemplateParameter<std::size_t>("RK0");
    auto rm1 = solution.GetTemplateParameter<std::size_t>("RM1");
    auto rn1 = solution.GetTemplateParameter<std::size_t>("RN1");
    auto rk1 = solution.GetTemplateParameter<std::size_t>("RK1");

    bool merge_heads = solution.GetTemplateParameter<std::string>("MergeNumHeadGroupsSeqLenQ") == "true";

    const std::size_t warp_size  = 64;
    const std::size_t num_warps  = std::max(rm0 * rn0 * rk0, rm1 * rn1 * rk1);
    const std::size_t block_size = num_warps * warp_size;

    // When kMergeNumHeadGroupsSeqLenQ is true, the kernel merges head groups with seqlen_q
    const std::size_t nhead_ratio = prob.nhead / prob.nhead_k;
    const std::size_t M_eff       = merge_heads ? prob.M * nhead_ratio : prob.M;
    const std::size_t nhead_eff   = merge_heads ? prob.nhead_k : prob.nhead;

    const auto grid_m = integer_divide_ceil(M_eff, bm0);
    const auto grid_n = integer_divide_ceil(prob.O, bn1);

    dim3 grid(grid_m * grid_n * prob.num_splits, nhead_eff, prob.batch);
    dim3 block(block_size, 1, 1);

    return {grid, block};
}

std::string make_splitkv_kernel_source(const ck::host::device_fmha_splitkv::Problem& prob,
                                       const ck::host::Solution& solution,
                                       const SplitKVParams& params)
{
    return ck::host::InterpolateString(
        ck::host::splitkv_kernel_template,
        {{"include", prob.GetIncludeHeader()},
         {"template", solution.ToTemplateString()},
         {"dtype", "ck_tile::fp16_t"},
         {"batch", std::to_string(params.batch)},
         {"nhead", std::to_string(params.nhead)},
         {"nhead_k", std::to_string(params.nhead_k)},
         {"m", std::to_string(params.M)},
         {"n", std::to_string(params.N)},
         {"k", std::to_string(params.K)},
         {"o", std::to_string(params.O)},
         {"num_splits", std::to_string(params.num_splits)},
         {"q_stride_batch", std::to_string(params.q_stride_batch)},
         {"q_stride_nhead", std::to_string(params.q_stride_nhead)},
         {"q_stride_m", std::to_string(params.q_stride_m)},
         {"k_stride_batch", std::to_string(params.k_stride_batch)},
         {"k_stride_nhead", std::to_string(params.k_stride_nhead)},
         {"k_stride_n", std::to_string(params.k_stride_n)},
         {"v_stride_batch", std::to_string(params.v_stride_batch)},
         {"v_stride_nhead", std::to_string(params.v_stride_nhead)},
         {"v_stride_n", std::to_string(params.v_stride_n)},
         {"o_acc_stride_batch", std::to_string(params.o_acc_stride_batch)},
         {"o_acc_stride_nhead", std::to_string(params.o_acc_stride_nhead)},
         {"o_acc_stride_split", std::to_string(params.o_acc_stride_split)},
         {"o_acc_stride_m", std::to_string(params.o_acc_stride_m)},
         {"lse_acc_stride_batch", std::to_string(params.lse_acc_stride_batch)},
         {"lse_acc_stride_nhead", std::to_string(params.lse_acc_stride_nhead)},
         {"lse_acc_stride_split", std::to_string(params.lse_acc_stride_split)},
         {"scale_s", std::to_string(params.scale_s) + "f"}});
}

inline void print_solution(const Solution& solution)
{
    std::cout << "  DataType:                   " << solution.GetTemplateParameter("DataType") << "\n"
              << "  BM0:                        " << solution.GetTemplateParameter("BM0") << "\n"
              << "  BN0:                        " << solution.GetTemplateParameter("BN0") << "\n"
              << "  BK0:                        " << solution.GetTemplateParameter("BK0") << "\n"
              << "  BN1:                        " << solution.GetTemplateParameter("BN1") << "\n"
              << "  BK1:                        " << solution.GetTemplateParameter("BK1") << "\n"
              << "  BK0Max:                     " << solution.GetTemplateParameter("BK0Max") << "\n"
              << "  RM0:                        " << solution.GetTemplateParameter("RM0") << "\n"
              << "  RN0:                        " << solution.GetTemplateParameter("RN0") << "\n"
              << "  RK0:                        " << solution.GetTemplateParameter("RK0") << "\n"
              << "  RM1:                        " << solution.GetTemplateParameter("RM1") << "\n"
              << "  RN1:                        " << solution.GetTemplateParameter("RN1") << "\n"
              << "  RK1:                        " << solution.GetTemplateParameter("RK1") << "\n"
              << "  WM0:                        " << solution.GetTemplateParameter("WM0") << "\n"
              << "  WN0:                        " << solution.GetTemplateParameter("WN0") << "\n"
              << "  WK0:                        " << solution.GetTemplateParameter("WK0") << "\n"
              << "  WM1:                        " << solution.GetTemplateParameter("WM1") << "\n"
              << "  WN1:                        " << solution.GetTemplateParameter("WN1") << "\n"
              << "  WK1:                        " << solution.GetTemplateParameter("WK1") << "\n"
              << "  IsVRowMajor:                " << solution.GetTemplateParameter("IsVRowMajor") << "\n"
              << "  PadM:                       " << solution.GetTemplateParameter("PadM") << "\n"
              << "  PadN:                       " << solution.GetTemplateParameter("PadN") << "\n"
              << "  PadK:                       " << solution.GetTemplateParameter("PadK") << "\n"
              << "  PadO:                       " << solution.GetTemplateParameter("PadO") << "\n"
              << "  HasUnevenSplits:            " << solution.GetTemplateParameter("HasUnevenSplits") << "\n"
              << "  MergeNumHeadGroupsSeqLenQ:  " << solution.GetTemplateParameter("MergeNumHeadGroupsSeqLenQ") << "\n"
              << "  PipelineTag:                " << solution.GetTemplateParameter("PipelineTag") << "\n"
              << std::flush;
}

} // namespace host
} // namespace ck
