// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_splitkv/problem.hpp"
#include "ck/host/device_fmha_splitkv/operation.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "ck/host/headers.hpp"
#include "common.hpp"
#include "fmha_fwd_splitkv_common.hpp"
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include <test.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

using half = _Float16;

TEST_CASE(test_splitkv_basic)
{
    ck::host::device_fmha_splitkv::Problem prob;
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

    const std::size_t q_size       = prob.batch * prob.nhead * prob.M * prob.K;
    const std::size_t k_size       = prob.batch * prob.nhead_k * prob.N * prob.K;
    const std::size_t v_size       = prob.batch * prob.nhead_k * prob.N * prob.O;
    const std::size_t o_acc_size   = prob.batch * prob.nhead * prob.num_splits * prob.M * prob.O;
    const std::size_t lse_acc_size = prob.batch * prob.nhead * prob.num_splits * prob.M;

    std::vector<float> q_data(q_size);
    std::vector<float> k_data(k_size);
    std::vector<float> v_data(v_size);
    std::generate(q_data.begin(), q_data.end(), [&]() { return dist(rng); });
    std::generate(k_data.begin(), k_data.end(), [&]() { return dist(rng); });
    std::generate(v_data.begin(), v_data.end(), [&]() { return dist(rng); });

    auto ref_params = ck::host::make_splitkv_params(prob, scale_s);

    const auto make_device_buff = [](const std::vector<float>& data) {
        rtc::buffer<half> host(data.size());
        std::transform(data.begin(), data.end(), host.begin(), [](float val) { return half(val); });
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
        srcs.push_back({"main.cpp", make_splitkv_kernel_source(prob, solution, ref_params)});

        rtc::compile_options options;
        options.kernel_name = "f";
        auto kernel         = rtc::compile_kernel(srcs, options);

        auto [grid, block] = ck::host::get_splitkv_launch_dims(solution, prob);

        std::cout << "  Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << "), "
                  << "Block: (" << block.x << ")" << std::endl;

        // Allocate output buffers
        rtc::buffer<float> o_acc_host(o_acc_size);
        rtc::buffer<float> lse_acc_host(lse_acc_size);
        std::fill(o_acc_host.begin(), o_acc_host.end(), 0.0f);
        std::fill(
            lse_acc_host.begin(), lse_acc_host.end(), -std::numeric_limits<float>::infinity());

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
