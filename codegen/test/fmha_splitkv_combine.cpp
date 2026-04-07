// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_splitkv_combine/problem.hpp"
#include "ck/host/device_fmha_splitkv_combine/operation.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "ck/host/headers.hpp"
#include "common.hpp"
#include "fmha_fwd_splitkv_combine_common.hpp"
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include <test.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

using half = _Float16;

TEST_CASE(test_combine_basic)
{
    ck::host::device_fmha_splitkv_combine::Problem prob;
    prob.M          = 1;  // seqlen_q (decode typically 1)
    prob.O          = 64; // hdim_v
    prob.batch      = 2;
    prob.nhead      = 4;
    prob.num_splits = 2;
    prob.dtype      = ck::host::DataType::Half;

    auto solutions = prob.GetSolutions("gfx90a");
    std::cout << "Number of Combine solutions: " << solutions.size() << std::endl;

    EXPECT(!solutions.empty());

    // Generate random input data (simulating SplitKV output)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::size_t lse_acc_size = prob.batch * prob.nhead * prob.num_splits * prob.M;
    const std::size_t o_acc_size   = prob.batch * prob.nhead * prob.num_splits * prob.M * prob.O;
    const std::size_t o_size       = prob.batch * prob.nhead * prob.M * prob.O;

    std::vector<float> lse_acc_data(lse_acc_size);
    std::vector<float> o_acc_data(o_acc_size);
    std::generate(lse_acc_data.begin(), lse_acc_data.end(), [&]() { return dist(rng); });
    std::generate(o_acc_data.begin(), o_acc_data.end(), [&]() { return dist(rng); });

    auto ref_params = ck::host::make_splitkv_combine_params(prob);

    rtc::buffer<float> lse_acc_host(lse_acc_size);
    rtc::buffer<float> o_acc_host(o_acc_size);
    std::copy(lse_acc_data.begin(), lse_acc_data.end(), lse_acc_host.begin());
    std::copy(o_acc_data.begin(), o_acc_data.end(), o_acc_host.begin());

    auto lse_acc_device = to_gpu(lse_acc_host);
    auto o_acc_device   = to_gpu(o_acc_host);

    for(std::size_t sol_idx = 0; sol_idx < solutions.size(); ++sol_idx)
    {
        auto&& solution = solutions[sol_idx];
        std::cout << "Testing solution " << (sol_idx + 1) << "/" << solutions.size() << ": "
                  << solution.ToTemplateString() << std::endl;

        auto srcs = get_tile_headers_for_test();
        srcs.push_back({"main.cpp", make_splitkv_combine_kernel_source(prob, solution, ref_params)});

        rtc::compile_options options;
        options.kernel_name = "f";
        auto kernel         = rtc::compile_kernel(srcs, options);

        auto [grid, block] = ck::host::get_splitkv_combine_launch_dims(solution, prob);

        std::cout << "  Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << "), "
                  << "Block: (" << block.x << ")" << std::endl;

        // Allocate output buffer
        rtc::buffer<half> o_host(o_size);
        std::fill(o_host.begin(), o_host.end(), half(0.0f));
        auto o_device = to_gpu(o_host);

        kernel.launch(nullptr, grid, block)(
            lse_acc_device.data(), o_acc_device.data(), o_device.data());

        o_host = rtc::from_gpu(o_device);

        // Basic sanity check: output should have finite values
        bool has_nan = false;
        for(auto v : o_host)
        {
            if(std::isnan(static_cast<float>(v)) || std::isinf(static_cast<float>(v)))
            {
                has_nan = true;
                break;
            }
        }
        EXPECT(!has_nan);

        std::cout << "  PASSED" << std::endl;
    }

    std::cout << "All " << solutions.size() << " Combine solutions executed successfully"
              << std::endl;
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
