// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_splitkv/problem.hpp"
#include "ck/host/device_fmha_splitkv/operation.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "ck/host/headers.hpp"
#include "common.hpp"
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include <test.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

using ck::host::Solution;
using ck::host::device_fmha_splitkv::Problem;

using half = _Float16;

const std::string kernel_template = R"__ck__(
#include <cmath>
#include <cstdint>
#include <cassert>
#include <sstream>
#include <${include}>

using KernelType = ${template};

extern "C" __launch_bounds__(KernelType::Kernel::kBlockSize, KernelType::Kernel::kBlockPerCu)
__global__ void f(const ${dtype}* q, const ${dtype}* k, const ${dtype}* v,
                  float* o_acc, float* lse_acc) {
    
    constexpr float scale_s = ${scale_s};
    
    using Kernel = KernelType;
    
    constexpr auto desc = Kernel::make_descriptor(
        // Q: [batch, nhead, M, K]
        ck_tile::make_tuple(${batch}, ${nhead}, ${m}, ${k}),
        ck_tile::make_tuple(${q_stride_batch}, ${q_stride_nhead}, ${q_stride_m}),
        // K: [batch, nhead_k, N, K]
        ck_tile::make_tuple(${batch}, ${nhead_k}, ${n}, ${k}),
        ck_tile::make_tuple(${k_stride_batch}, ${k_stride_nhead}, ${k_stride_n}),
        // V: [batch, nhead_k, N, O]
        ck_tile::make_tuple(${batch}, ${nhead_k}, ${n}, ${o}),
        ck_tile::make_tuple(${v_stride_batch}, ${v_stride_nhead}, ${v_stride_n}),
        // O_acc dims: [batch, nhead, num_splits, M, O] - num_splits extracted from index 2
        ck_tile::make_tuple(${batch}, ${nhead}, ${num_splits}, ${m}, ${o}),
        // O_acc strides: [batch, nhead, split, m]
        ck_tile::make_tuple(${o_acc_stride_batch}, ${o_acc_stride_nhead}, ${o_acc_stride_split}, ${o_acc_stride_m}),
        // LSE_acc strides: [batch, nhead, split]
        ck_tile::make_tuple(${lse_acc_stride_batch}, ${lse_acc_stride_nhead}, ${lse_acc_stride_split}));
    
    static_assert(desc.IsValid(), "Invalid SplitKV kernel configuration");
    
    Kernel::Run(desc, scale_s, q, k, v, lse_acc, o_acc);
}
)__ck__";

struct SplitKVRefParams
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

std::string make_kernel_source(const Problem& prob,
                               const Solution& solution,
                               const SplitKVRefParams& params)
{
    return ck::host::InterpolateString(
        kernel_template,
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

SplitKVRefParams make_ref_params(const Problem& prob, float scale_s)
{
    SplitKVRefParams p;
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

std::pair<dim3, dim3> get_launch_dims(const Solution& solution, const Problem& prob)
{
    // Block tile sizes
    auto bm0 = solution.GetTemplateParameter<std::size_t>("BM0");
    auto bn1 = solution.GetTemplateParameter<std::size_t>("BN1");

    // Block warps for Gemm0 - sequence<RM0, RN0, RK0>
    auto rm0 = solution.GetTemplateParameter<std::size_t>("RM0");
    auto rn0 = solution.GetTemplateParameter<std::size_t>("RN0");
    auto rk0 = solution.GetTemplateParameter<std::size_t>("RK0");

    // Block warps for Gemm1 - sequence<RM1, RN1, RK1>
    auto rm1 = solution.GetTemplateParameter<std::size_t>("RM1");
    auto rn1 = solution.GetTemplateParameter<std::size_t>("RN1");
    auto rk1 = solution.GetTemplateParameter<std::size_t>("RK1");

    const std::size_t warp_size  = 64;
    const std::size_t num_warps  = std::max(rm0 * rn0 * rk0, rm1 * rn1 * rk1);
    const std::size_t block_size = num_warps * warp_size;

    // SplitKV grid from FmhaFwdSplitKVKernel::GridSize:
    // x = ceil(seqlen_q / kM0) * ceil(hdim_v / kN1) * num_splits
    // y = nhead
    // z = batch
    const auto grid_m = ck::host::integer_divide_ceil(prob.M, bm0);
    const auto grid_n = ck::host::integer_divide_ceil(prob.O, bn1);

    dim3 grid(grid_m * grid_n * prob.num_splits, prob.nhead, prob.batch);
    dim3 block(block_size, 1, 1);

    return {grid, block};
}

TEST_CASE(test_splitkv_basic)
{
    Problem prob;
    prob.M             = 1;   // seqlen_q (decode typically 1)
    prob.N             = 128; // seqlen_k (KV cache length)
    prob.K             = 64;  // hdim_q
    prob.O             = 64;  // hdim_v
    prob.batch         = 2;
    prob.nhead         = 4;
    prob.nhead_k       = 4;
    prob.num_splits    = 2;
    prob.dtype         = ck::host::DataType::Half;
    prob.is_v_rowmajor = true;

    const float scale_s = 1.0f / std::sqrt(static_cast<float>(prob.K));

    auto solutions = prob.GetSolutions("gfx90a");
    std::cout << "Number of SplitKV solutions: " << solutions.size() << std::endl;

    EXPECT(!solutions.empty());

    // Generate random input data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const std::size_t q_size = prob.batch * prob.nhead * prob.M * prob.K;
    const std::size_t k_size = prob.batch * prob.nhead_k * prob.N * prob.K;
    const std::size_t v_size = prob.batch * prob.nhead_k * prob.N * prob.O;
    const std::size_t o_acc_size =
        prob.batch * prob.nhead * prob.num_splits * prob.M * prob.O;
    const std::size_t lse_acc_size = prob.batch * prob.nhead * prob.num_splits * prob.M;

    std::vector<float> q_data(q_size);
    std::vector<float> k_data(k_size);
    std::vector<float> v_data(v_size);
    std::generate(q_data.begin(), q_data.end(), [&]() { return dist(rng); });
    std::generate(k_data.begin(), k_data.end(), [&]() { return dist(rng); });
    std::generate(v_data.begin(), v_data.end(), [&]() { return dist(rng); });

    auto ref_params = make_ref_params(prob, scale_s);

    const auto make_device_buff = [](const std::vector<float>& data) {
        rtc::buffer<half> host(data.size());
        std::transform(
            data.begin(), data.end(), host.begin(), [](float val) { return half(val); });
        return to_gpu(host);
    };
    auto q_device = make_device_buff(q_data);
    auto k_device = make_device_buff(k_data);
    auto v_device = make_device_buff(v_data);

    for(std::size_t sol_idx = 0; sol_idx < solutions.size(); ++sol_idx)
    {
        auto&& solution = solutions[sol_idx];
        std::cout << "Testing solution " << (sol_idx + 1) << "/" << solutions.size() << ": "
                  << solution.ToTemplateString() << std::endl;

        auto srcs = get_tile_headers_for_test();
        srcs.push_back({"main.cpp", make_kernel_source(prob, solution, ref_params)});

        rtc::compile_options options;
        options.kernel_name = "f";
        auto kernel         = rtc::compile_kernel(srcs, options);

        auto [grid, block] = get_launch_dims(solution, prob);

        std::cout << "  Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << "), "
                  << "Block: (" << block.x << ")" << std::endl;

        // Allocate output buffers
        rtc::buffer<float> o_acc_host(o_acc_size);
        rtc::buffer<float> lse_acc_host(lse_acc_size);
        std::fill(o_acc_host.begin(), o_acc_host.end(), 0.0f);
        std::fill(lse_acc_host.begin(), lse_acc_host.end(), -std::numeric_limits<float>::infinity());

        auto o_acc_device   = to_gpu(o_acc_host);
        auto lse_acc_device = to_gpu(lse_acc_host);

        kernel.launch(nullptr, grid, block)(q_device.data(),
                                            k_device.data(),
                                            v_device.data(),
                                            lse_acc_device.data(),
                                            o_acc_device.data());

        o_acc_host   = rtc::from_gpu(o_acc_device);
        lse_acc_host = rtc::from_gpu(lse_acc_device);

        // Basic sanity check: o_acc and lse_acc should have finite values
        bool has_nan = false;
        for(auto v : o_acc_host)
        {
            if(std::isnan(v) || std::isinf(v))
            {
                has_nan = true;
                break;
            }
        }
        EXPECT(!has_nan);

        for(auto v : lse_acc_host)
        {
            if(std::isnan(v))
            {
                has_nan = true;
                break;
            }
        }
        EXPECT(!has_nan);

        std::cout << "  PASSED" << std::endl;
    }

    std::cout << "All " << solutions.size() << " SplitKV solutions executed successfully"
              << std::endl;
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
