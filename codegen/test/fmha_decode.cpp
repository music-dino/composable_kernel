// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test for Flash Decoding: SplitKV + Combine kernels

#include "ck/host/device_fmha_splitkv/problem.hpp"
#include "ck/host/device_fmha_splitkv/operation.hpp"
#include "ck/host/device_fmha_splitkv_combine/problem.hpp"
#include "ck/host/device_fmha_splitkv_combine/operation.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "ck/host/headers.hpp"
#include "common.hpp"
#include "fmha_fwd_ref.hpp"
#include "fmha_fwd_splitkv_common.hpp"
#include "fmha_fwd_splitkv_combine_common.hpp"
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include <test.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace splitkv = ck::host::device_fmha_splitkv;
namespace combine = ck::host::device_fmha_splitkv_combine;
using half        = _Float16;

TEST_CASE(test_decode_splitkv_combine)
{
    // SplitKV problem
    splitkv::Problem splitkv_prob;
    splitkv_prob.M             = 1;   // seqlen_q (decode)
    splitkv_prob.N             = 128; // seqlen_k
    splitkv_prob.K             = 64;  // hdim_q
    splitkv_prob.O             = 64;  // hdim_v
    splitkv_prob.batch         = 2;
    splitkv_prob.nhead         = 4;
    splitkv_prob.nhead_k       = 4;
    splitkv_prob.num_splits    = 2;
    splitkv_prob.dtype         = ck::host::DataType::Half;
    splitkv_prob.is_v_rowmajor = true;

    // Combine problem (shares dimensions with SplitKV)
    combine::Problem combine_prob;
    combine_prob.M          = splitkv_prob.M;
    combine_prob.O          = splitkv_prob.O;
    combine_prob.batch      = splitkv_prob.batch;
    combine_prob.nhead      = splitkv_prob.nhead;
    combine_prob.num_splits = splitkv_prob.num_splits;
    combine_prob.dtype      = ck::host::DataType::Half;

    const float scale_s = 1.0f / std::sqrt(static_cast<float>(splitkv_prob.K));

    auto splitkv_solutions = splitkv_prob.GetSolutions("gfx90a");
    auto combine_solutions = combine_prob.GetSolutions("gfx90a");

    std::cout << "SplitKV solutions: " << splitkv_solutions.size() << std::endl;
    std::cout << "Combine solutions: " << combine_solutions.size() << std::endl;

    EXPECT(!splitkv_solutions.empty());
    EXPECT(!combine_solutions.empty());

    // Generate random input data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const std::size_t q_size =
        splitkv_prob.batch * splitkv_prob.nhead * splitkv_prob.M * splitkv_prob.K;
    const std::size_t k_size =
        splitkv_prob.batch * splitkv_prob.nhead_k * splitkv_prob.N * splitkv_prob.K;
    const std::size_t v_size =
        splitkv_prob.batch * splitkv_prob.nhead_k * splitkv_prob.N * splitkv_prob.O;
    const std::size_t o_acc_size = splitkv_prob.batch * splitkv_prob.nhead *
                                   splitkv_prob.num_splits * splitkv_prob.M * splitkv_prob.O;
    const std::size_t lse_acc_size =
        splitkv_prob.batch * splitkv_prob.nhead * splitkv_prob.num_splits * splitkv_prob.M;
    const std::size_t o_size =
        splitkv_prob.batch * splitkv_prob.nhead * splitkv_prob.M * splitkv_prob.O;

    std::vector<float> q_data(q_size);
    std::vector<float> k_data(k_size);
    std::vector<float> v_data(v_size);
    std::generate(q_data.begin(), q_data.end(), [&]() { return dist(rng); });
    std::generate(k_data.begin(), k_data.end(), [&]() { return dist(rng); });
    std::generate(v_data.begin(), v_data.end(), [&]() { return dist(rng); });

    auto splitkv_params = ck::host::make_splitkv_params(splitkv_prob, scale_s);
    auto combine_params = ck::host::make_splitkv_combine_params(combine_prob);

    // Convert to fp16 and upload to GPU
    const auto make_device_buff = [](const std::vector<float>& data) {
        rtc::buffer<half> host(data.size());
        std::transform(data.begin(), data.end(), host.begin(), [](float val) { return half(val); });
        return to_gpu(host);
    };
    auto q_device = make_device_buff(q_data);
    auto k_device = make_device_buff(k_data);
    auto v_device = make_device_buff(v_data);

    // Use first solution for each kernel
    auto& splitkv_sol = splitkv_solutions[0];
    auto& combine_sol = combine_solutions[0];

    std::cout << "SplitKV: " << splitkv_sol.ToTemplateString() << std::endl;
    std::cout << "Combine: " << combine_sol.ToTemplateString() << std::endl;

    // Compile SplitKV kernel
    auto splitkv_srcs = get_tile_headers_for_test();
    splitkv_srcs.push_back(
        {"main.cpp", make_splitkv_kernel_source(splitkv_prob, splitkv_sol, splitkv_params)});
    rtc::compile_options splitkv_opts;
    splitkv_opts.kernel_name = "f";
    auto splitkv_kernel      = rtc::compile_kernel(splitkv_srcs, splitkv_opts);

    // Compile Combine kernel
    auto combine_srcs = get_tile_headers_for_test();
    combine_srcs.push_back(
        {"main.cpp",
         make_splitkv_combine_kernel_source(combine_prob, combine_sol, combine_params)});
    rtc::compile_options combine_opts;
    combine_opts.kernel_name = "f";
    auto combine_kernel      = rtc::compile_kernel(combine_srcs, combine_opts);

    // Allocate intermediate and output buffers
    rtc::buffer<float> o_acc_host(o_acc_size);
    rtc::buffer<float> lse_acc_host(lse_acc_size);
    std::fill(o_acc_host.begin(), o_acc_host.end(), 0.0f);
    std::fill(lse_acc_host.begin(), lse_acc_host.end(), -std::numeric_limits<float>::infinity());

    auto o_acc_device   = to_gpu(o_acc_host);
    auto lse_acc_device = to_gpu(lse_acc_host);

    rtc::buffer<half> o_host(o_size);
    std::fill(o_host.begin(), o_host.end(), half(0.0f));
    auto o_device = to_gpu(o_host);

    // Launch SplitKV
    auto [splitkv_grid, splitkv_block] =
        ck::host::get_splitkv_launch_dims(splitkv_sol, splitkv_prob);
    std::cout << "SplitKV Grid: (" << splitkv_grid.x << ", " << splitkv_grid.y << ", "
              << splitkv_grid.z << "), Block: (" << splitkv_block.x << ")" << std::endl;

    splitkv_kernel.launch(nullptr, splitkv_grid, splitkv_block)(q_device.data(),
                                                                k_device.data(),
                                                                v_device.data(),
                                                                o_acc_device.data(),
                                                                lse_acc_device.data());

    auto [combine_grid, combine_block] =
        ck::host::get_splitkv_combine_launch_dims(combine_sol, combine_prob);
    std::cout << "Combine Grid: (" << combine_grid.x << ", " << combine_grid.y << ", "
              << combine_grid.z << "), Block: (" << combine_block.x << ")" << std::endl;

    combine_kernel.launch(nullptr, combine_grid, combine_block)(
        lse_acc_device.data(), o_acc_device.data(), o_device.data());

    // Read back output
    o_host = rtc::from_gpu(o_device);

    // Convert GPU output to float for comparison
    std::vector<float> result(o_size);
    std::transform(
        o_host.begin(), o_host.end(), result.begin(), [](half v) { return static_cast<float>(v); });

    // Compute reference output using CPU implementation
    ck::host::device_fmha_fwd::FmhaFwdRefParams ref_params;
    ref_params.batch          = splitkv_params.batch;
    ref_params.nhead          = splitkv_params.nhead;
    ref_params.nhead_k        = splitkv_params.nhead_k;
    ref_params.M              = splitkv_params.M;
    ref_params.N              = splitkv_params.N;
    ref_params.K              = splitkv_params.K;
    ref_params.O              = splitkv_params.O;
    ref_params.scale_s        = splitkv_params.scale_s;
    ref_params.q_stride_batch = splitkv_params.q_stride_batch;
    ref_params.q_stride_nhead = splitkv_params.q_stride_nhead;
    ref_params.q_stride_m     = splitkv_params.q_stride_m;
    ref_params.k_stride_batch = splitkv_params.k_stride_batch;
    ref_params.k_stride_nhead = splitkv_params.k_stride_nhead;
    ref_params.k_stride_n     = splitkv_params.k_stride_n;
    ref_params.v_stride_batch = splitkv_params.v_stride_batch;
    ref_params.v_stride_nhead = splitkv_params.v_stride_nhead;
    ref_params.v_stride_n     = splitkv_params.v_stride_n;
    ref_params.o_stride_batch = combine_params.o_stride_batch;
    ref_params.o_stride_nhead = combine_params.o_stride_nhead;
    ref_params.o_stride_m     = combine_params.o_stride_m;

    std::vector<float> o_ref(o_size);
    ck::host::device_fmha_fwd::cpu_attention_ref(q_data, k_data, v_data, o_ref, ref_params);

    // Compare outputs using allclose (same as fmha_fwd tests)
    CHECK(allclose(result, o_ref, 0.0001, 0.0001));

    std::cout << "Flash Decoding (SplitKV + Combine) PASSED" << std::endl;
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
