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
#include <cmath>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

namespace splitkv = ck::host::device_fmha_splitkv;
namespace combine = ck::host::device_fmha_splitkv_combine;
using half        = _Float16;

struct VectorStats
{
    double min;
    double max;
    double mean;
    double stdev;
    double stderr;
};

VectorStats compute_stats(const std::vector<float>& v)
{
    VectorStats stats{};
    if(v.empty())
        return stats;

    double min_val = v[0];
    double max_val = v[0];
    double sum     = 0.0;
    double sum_sq  = 0.0;

    for(float val : v)
    {
        double d = static_cast<double>(val);
        min_val  = std::min(min_val, d);
        max_val  = std::max(max_val, d);
        sum += d;
        sum_sq += d * d;
    }

    double n     = static_cast<double>(v.size());
    stats.min    = min_val;
    stats.max    = max_val;
    stats.mean   = sum / n;
    stats.stdev  = std::sqrt((sum_sq / n) - (stats.mean * stats.mean));
    stats.stderr = stats.stdev / std::sqrt(n);

    return stats;
}

void print_stats(const std::string& name, const VectorStats& stats)
{
    std::cout << name << ": min=" << stats.min << ", max=" << stats.max << ", mean=" << stats.mean
              << ", stdev=" << stats.stdev << ", stderr=" << stats.stderr << std::endl;
}

TEST_CASE(test_decode_splitkv_combine)
{
    constexpr auto test_body = [](const splitkv::Problem& splitkv_prob) {
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

        ck::host::SplitKVParams splitkv_params =
            ck::host::make_splitkv_params(splitkv_prob, scale_s);
        ck::host::SplitKVCombineParams combine_params =
            ck::host::make_splitkv_combine_params(combine_prob);

        // Convert to fp16 and upload to GPU
        const auto make_device_buff = [](const std::vector<float>& data) {
            rtc::buffer<half> host(data.size());
            std::transform(
                data.begin(), data.end(), host.begin(), [](float val) { return half(val); });
            return to_gpu(host);
        };
        auto q_device = make_device_buff(q_data);
        auto k_device = make_device_buff(k_data);
        auto v_device = make_device_buff(v_data);

        // Compute reference output once (same for all solutions)
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
        print_stats("Reference", compute_stats(o_ref));

        // Compile Combine kernel once (typically only 1 solution)
        ck::host::Solution& combine_sol = combine_solutions[0];
        std::cout << "Combine: " << combine_sol.ToTemplateString() << std::endl;

        auto combine_srcs = get_tile_headers_for_test();
        combine_srcs.push_back(
            {"main.cpp",
             make_splitkv_combine_kernel_source(combine_prob, combine_sol, combine_params)});
        rtc::compile_options combine_opts;
        combine_opts.kernel_name = "f";
        auto combine_kernel      = rtc::compile_kernel(combine_srcs, combine_opts);

        auto [combine_grid, combine_block] =
            ck::host::get_splitkv_combine_launch_dims(combine_sol, combine_prob);

        // Test each SplitKV solution
        for(std::size_t sol_idx = 0; sol_idx < splitkv_solutions.size(); ++sol_idx)
        {
            ck::host::Solution& splitkv_sol = splitkv_solutions[sol_idx];
            std::cout << "\n=== SplitKV solution " << (sol_idx + 1) << "/"
                      << splitkv_solutions.size() << " ===" << std::endl;
            std::cout << "SplitKV: " << splitkv_sol.ToTemplateString() << std::endl;

            // Compile SplitKV kernel
            auto splitkv_srcs = get_tile_headers_for_test();
            splitkv_srcs.push_back(
                {"main.cpp",
                 make_splitkv_kernel_source(splitkv_prob, splitkv_sol, splitkv_params)});
            rtc::compile_options splitkv_opts;
            splitkv_opts.kernel_name = "f";
            try
            {
                auto splitkv_kernel = rtc::compile_kernel(splitkv_srcs, splitkv_opts);

                // Allocate and initialize intermediate buffers
                rtc::buffer<float> o_acc_host(o_acc_size);
                rtc::buffer<float> lse_acc_host(lse_acc_size);
                std::fill(o_acc_host.begin(), o_acc_host.end(), 0.0f);
                std::fill(lse_acc_host.begin(),
                          lse_acc_host.end(),
                          -std::numeric_limits<float>::infinity());

                auto o_acc_device   = to_gpu(o_acc_host);
                auto lse_acc_device = to_gpu(lse_acc_host);

                rtc::buffer<half> o_host(o_size);
                std::fill(o_host.begin(), o_host.end(), half(0.0f));
                auto o_device = to_gpu(o_host);

                // Launch SplitKV
                auto [splitkv_grid, splitkv_block] =
                    ck::host::get_splitkv_launch_dims(splitkv_sol, splitkv_prob);

                splitkv_kernel.launch(nullptr, splitkv_grid, splitkv_block)(q_device.data(),
                                                                            k_device.data(),
                                                                            v_device.data(),
                                                                            o_acc_device.data(),
                                                                            lse_acc_device.data());

                // Launch Combine
                combine_kernel.launch(nullptr, combine_grid, combine_block)(
                    lse_acc_device.data(), o_acc_device.data(), o_device.data());

                // Read back output
                o_host = rtc::from_gpu(o_device);

                // Convert GPU output to float for comparison
                std::vector<float> result(o_size);
                std::transform(o_host.begin(), o_host.end(), result.begin(), [](half v) {
                    return static_cast<float>(v);
                });

                print_stats("GPU Result", compute_stats(result));

                auto ver_ref = allclose(result, o_ref, 0.0001, 0.0001);
                CHECK(ver_ref);
                std::cout << "Solution " << (sol_idx + 1) << (ver_ref ? " PASSED" : " FAILED")
                          << std::endl;
                if(!ver_ref)
                {
                    print_solution(splitkv_sol);
                }
            }
            catch(const std::exception& e)
            {
                CHECK(false);
                std::cout << "Solution " << (sol_idx + 1) << " FAILED COMPILATION: " << e.what()
                          << std::endl;
                print_solution(splitkv_sol);
            }
        }

        // std::cout << "\nFlash Decoding (SplitKV + Combine) - All " << splitkv_solutions.size()
        //           << " solutions PASSED" << std::endl;
    };

    struct TestConfig
    {
        std::size_t M;
        std::size_t N;
        std::size_t K;
        std::size_t O;
        std::size_t nhead;
        std::size_t nhead_k;
        std::size_t num_splits;
    };

    std::vector<TestConfig> configs = {
        // Regular MHA (nhead == nhead_k), M=1 (decode)
        {1, 128, 64, 64, 4, 4, 2},
        {1, 256, 64, 64, 4, 4, 4},
        {1, 512, 128, 128, 4, 4, 4},
        {1, 128, 32, 32, 4, 4, 2},
        {1, 64, 64, 64, 4, 4, 2},
        {1, 2048, 64, 64, 4, 4, 8},

        // GQA (nhead_k < nhead), M=1 (decode)
        {1, 128, 64, 64, 8, 2, 2},
        {1, 256, 64, 64, 8, 2, 4},
        {1, 512, 128, 128, 8, 4, 4},
        {1, 128, 64, 64, 8, 1, 2},    // nhead_k=1 (extreme GQA)
        {1, 256, 128, 128, 16, 4, 4}, // larger nhead ratio

        // Different num_splits, M=1 (decode)
        {1, 256, 64, 64, 4, 4, 2},
        {1, 256, 64, 64, 4, 4, 8},
        {1, 1024, 64, 64, 4, 4, 8},
        {1, 512, 64, 64, 4, 4, 16}, // many splits

        // M > 1 (prefill-like scenarios)
        {8, 256, 64, 64, 4, 4, 2},
        {16, 512, 64, 64, 4, 4, 4},
        {32, 256, 128, 128, 4, 4, 4},
        {64, 512, 64, 64, 4, 4, 4},
        {4, 128, 64, 64, 4, 4, 2},
        {128, 256, 64, 64, 4, 4, 4}, // larger M
        {32, 1024, 64, 64, 4, 4, 8}, // larger N with M > 1

        // M > 1 with GQA
        {8, 256, 64, 64, 8, 2, 2},
        {16, 512, 128, 128, 8, 4, 4},
        {4, 128, 64, 64, 8, 1, 2},    // GQA with nhead_k=1, M > 1
        {32, 256, 128, 128, 8, 2, 4}, // larger M with GQA

        // Edge cases: non-power-of-2 and uneven dimensions
        {1, 100, 64, 64, 4, 4, 2},  // N not power of 2
        {1, 300, 64, 64, 4, 4, 4},  // N not power of 2
        {7, 256, 64, 64, 4, 4, 2},  // M not power of 2
        {13, 512, 64, 64, 4, 4, 4}, // M prime number

        // Different hdim combinations
        {1, 256, 32, 32, 4, 4, 2},
        {1, 256, 96, 128, 4, 4, 4}, // K != O
        {8, 512, 32, 32, 4, 4, 4},
        {16, 256, 128, 128, 4, 4, 2},
    };

    for(std::size_t cfg_idx = 0; cfg_idx < configs.size(); ++cfg_idx)
    {
        const TestConfig& cfg = configs[cfg_idx];

        std::cout << "\n========================================" << std::endl;
        std::cout << "Config " << (cfg_idx + 1) << "/" << configs.size() << ": "
                  << "M=" << cfg.M << ", N=" << cfg.N << ", K=" << cfg.K << ", O=" << cfg.O
                  << ", nhead=" << cfg.nhead << ", nhead_k=" << cfg.nhead_k
                  << ", num_splits=" << cfg.num_splits << std::endl;
        std::cout << "========================================" << std::endl;

        splitkv::Problem splitkv_prob;
        splitkv_prob.M             = cfg.M;
        splitkv_prob.N             = cfg.N;
        splitkv_prob.K             = cfg.K;
        splitkv_prob.O             = cfg.O;
        splitkv_prob.batch         = 2;
        splitkv_prob.nhead         = cfg.nhead;
        splitkv_prob.nhead_k       = cfg.nhead_k;
        splitkv_prob.num_splits    = cfg.num_splits;
        splitkv_prob.dtype         = ck::host::DataType::Half;
        splitkv_prob.is_v_rowmajor = true;

        test_body(splitkv_prob);
    }

    // std::cout << "\n========================================" << std::endl;
    // std::cout << "All " << configs.size() << " configurations PASSED" << std::endl;
    // std::cout << "========================================" << std::endl;
}

TEST_CASE(test_decode_splitkv_combine2)
{
    constexpr auto test_body = [](const splitkv::Problem& splitkv_prob) {
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

        ck::host::SplitKVParams splitkv_params =
            ck::host::make_splitkv_params(splitkv_prob, scale_s);
        ck::host::SplitKVCombineParams combine_params =
            ck::host::make_splitkv_combine_params(combine_prob);

        // Convert to fp16 and upload to GPU
        const auto make_device_buff = [](const std::vector<float>& data) {
            rtc::buffer<half> host(data.size());
            std::transform(
                data.begin(), data.end(), host.begin(), [](float val) { return half(val); });
            return to_gpu(host);
        };
        auto q_device = make_device_buff(q_data);
        auto k_device = make_device_buff(k_data);
        auto v_device = make_device_buff(v_data);

        // Compute reference output once (same for all solutions)
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
        print_stats("Reference", compute_stats(o_ref));

        // Compile SplitKV kernel once
        ck::host::Solution& splitkv_sol = splitkv_solutions[0];
        // Compile SplitKV kernel
        auto splitkv_srcs = get_tile_headers_for_test();
        splitkv_srcs.push_back(
            {"main.cpp", make_splitkv_kernel_source(splitkv_prob, splitkv_sol, splitkv_params)});
        rtc::compile_options splitkv_opts;
        splitkv_opts.kernel_name = "f";
        auto splitkv_kernel      = rtc::compile_kernel(splitkv_srcs, splitkv_opts);
        // Launch SplitKV
        auto [splitkv_grid, splitkv_block] =
            ck::host::get_splitkv_launch_dims(splitkv_sol, splitkv_prob);

        // Test each SplitKV solution
        for(std::size_t sol_idx = 0; sol_idx < combine_solutions.size(); ++sol_idx)
        {
            std::cout << "\n=== Combine solution " << (sol_idx + 1) << "/"
                      << combine_solutions.size() << " ===" << std::endl;
            std::cout << "SplitKV: " << splitkv_sol.ToTemplateString() << std::endl;

            // Compile Combine kernel once (typically only 1 solution)
            ck::host::Solution& combine_sol = combine_solutions[sol_idx];
            std::cout << "Combine: " << combine_sol.ToTemplateString() << std::endl;

            auto combine_srcs = get_tile_headers_for_test();
            combine_srcs.push_back(
                {"main.cpp",
                 make_splitkv_combine_kernel_source(combine_prob, combine_sol, combine_params)});
            rtc::compile_options combine_opts;
            combine_opts.kernel_name = "f";

            try
            {
                auto combine_kernel = rtc::compile_kernel(combine_srcs, combine_opts);

                auto [combine_grid, combine_block] =
                    ck::host::get_splitkv_combine_launch_dims(combine_sol, combine_prob);
                // Allocate and initialize intermediate buffers
                rtc::buffer<float> o_acc_host(o_acc_size);
                rtc::buffer<float> lse_acc_host(lse_acc_size);
                std::fill(o_acc_host.begin(), o_acc_host.end(), 0.0f);
                std::fill(lse_acc_host.begin(),
                          lse_acc_host.end(),
                          -std::numeric_limits<float>::infinity());

                auto o_acc_device   = to_gpu(o_acc_host);
                auto lse_acc_device = to_gpu(lse_acc_host);

                rtc::buffer<half> o_host(o_size);
                std::fill(o_host.begin(), o_host.end(), half(0.0f));
                auto o_device = to_gpu(o_host);

                splitkv_kernel.launch(nullptr, splitkv_grid, splitkv_block)(q_device.data(),
                                                                            k_device.data(),
                                                                            v_device.data(),
                                                                            o_acc_device.data(),
                                                                            lse_acc_device.data());

                // Launch Combine
                combine_kernel.launch(nullptr, combine_grid, combine_block)(
                    lse_acc_device.data(), o_acc_device.data(), o_device.data());

                // Read back output
                o_host = rtc::from_gpu(o_device);

                // Convert GPU output to float for comparison
                std::vector<float> result(o_size);
                std::transform(o_host.begin(), o_host.end(), result.begin(), [](half v) {
                    return static_cast<float>(v);
                });

                print_stats("GPU Result", compute_stats(result));

                auto ver_ref = allclose(result, o_ref, 0.0001, 0.0001);
                CHECK(ver_ref);
                std::cout << "Solution " << (sol_idx + 1) << (ver_ref ? " PASSED" : " FAILED")
                          << std::endl;
                if(!ver_ref)
                {
                    print_solution(splitkv_sol);
                }
            }
            catch(const std::exception& e)
            {
                CHECK(false);
                std::cout << "Solution " << (sol_idx + 1) << " FAILED COMPILATION: " << e.what()
                          << std::endl;
                print_solution(splitkv_sol);
            }
        }

        // std::cout << "\nFlash Decoding (SplitKV + Combine) - All " << splitkv_solutions.size()
        //           << " solutions PASSED" << std::endl;
    };

    struct TestConfig
    {
        std::size_t M;
        std::size_t N;
        std::size_t K;
        std::size_t O;
        std::size_t nhead;
        std::size_t nhead_k;
        std::size_t num_splits;
    };

    std::vector<TestConfig> configs = {
        // Regular MHA (nhead == nhead_k), M=1 (decode)
        {1, 128, 64, 64, 4, 4, 2},
        {1, 256, 64, 64, 4, 4, 4},
        {1, 512, 128, 128, 4, 4, 4},
        {1, 128, 32, 32, 4, 4, 2},
        {1, 64, 64, 64, 4, 4, 2},
        {1, 2048, 64, 64, 4, 4, 8},

        // GQA (nhead_k < nhead), M=1 (decode)
        {1, 128, 64, 64, 8, 2, 2},
        {1, 256, 64, 64, 8, 2, 4},
        {1, 512, 128, 128, 8, 4, 4},
        {1, 128, 64, 64, 8, 1, 2},    // nhead_k=1 (extreme GQA)
        {1, 256, 128, 128, 16, 4, 4}, // larger nhead ratio

        // Different num_splits, M=1 (decode)
        {1, 256, 64, 64, 4, 4, 2},
        {1, 256, 64, 64, 4, 4, 8},
        {1, 1024, 64, 64, 4, 4, 8},
        {1, 512, 64, 64, 4, 4, 16}, // many splits

        // M > 1 (prefill-like scenarios)
        {8, 256, 64, 64, 4, 4, 2},
        {16, 512, 64, 64, 4, 4, 4},
        {32, 256, 128, 128, 4, 4, 4},
        {64, 512, 64, 64, 4, 4, 4},
        {4, 128, 64, 64, 4, 4, 2},
        {128, 256, 64, 64, 4, 4, 4}, // larger M
        {32, 1024, 64, 64, 4, 4, 8}, // larger N with M > 1

        // M > 1 with GQA
        {8, 256, 64, 64, 8, 2, 2},
        {16, 512, 128, 128, 8, 4, 4},
        {4, 128, 64, 64, 8, 1, 2},    // GQA with nhead_k=1, M > 1
        {32, 256, 128, 128, 8, 2, 4}, // larger M with GQA

        // Edge cases: non-power-of-2 and uneven dimensions
        {1, 100, 64, 64, 4, 4, 2},  // N not power of 2
        {1, 300, 64, 64, 4, 4, 4},  // N not power of 2
        {7, 256, 64, 64, 4, 4, 2},  // M not power of 2
        {13, 512, 64, 64, 4, 4, 4}, // M prime number

        // Different hdim combinations
        {1, 256, 32, 32, 4, 4, 2},
        {1, 256, 96, 128, 4, 4, 4}, // K != O
        {8, 512, 32, 32, 4, 4, 4},
        {16, 256, 128, 128, 4, 4, 2},
    };

    for(std::size_t cfg_idx = 0; cfg_idx < configs.size(); ++cfg_idx)
    {
        const TestConfig& cfg = configs[cfg_idx];

        std::cout << "\n========================================" << std::endl;
        std::cout << "Config " << (cfg_idx + 1) << "/" << configs.size() << ": "
                  << "M=" << cfg.M << ", N=" << cfg.N << ", K=" << cfg.K << ", O=" << cfg.O
                  << ", nhead=" << cfg.nhead << ", nhead_k=" << cfg.nhead_k
                  << ", num_splits=" << cfg.num_splits << std::endl;
        std::cout << "========================================" << std::endl;

        splitkv::Problem splitkv_prob;
        splitkv_prob.M             = cfg.M;
        splitkv_prob.N             = cfg.N;
        splitkv_prob.K             = cfg.K;
        splitkv_prob.O             = cfg.O;
        splitkv_prob.batch         = 2;
        splitkv_prob.nhead         = cfg.nhead;
        splitkv_prob.nhead_k       = cfg.nhead_k;
        splitkv_prob.num_splits    = cfg.num_splits;
        splitkv_prob.dtype         = ck::host::DataType::Half;
        splitkv_prob.is_v_rowmajor = true;

        test_body(splitkv_prob);
    }

    // std::cout << "\n========================================" << std::endl;
    // std::cout << "All " << configs.size() << " configurations PASSED" << std::endl;
    // std::cout << "========================================" << std::endl;
}

TEST_CASE(benchmark_fmha_decode)
{
    splitkv::Problem splitkv_prob;
    splitkv_prob.M             = 1;
    splitkv_prob.N             = 4096;
    splitkv_prob.K             = 96;
    splitkv_prob.O             = 64;
    splitkv_prob.batch         = 2;
    splitkv_prob.nhead         = 4;
    splitkv_prob.nhead_k       = 4;
    splitkv_prob.num_splits    = 2;
    splitkv_prob.dtype         = ck::host::DataType::Half;
    splitkv_prob.is_v_rowmajor = true;

    combine::Problem combine_prob;
    combine_prob.M          = splitkv_prob.M;
    combine_prob.O          = splitkv_prob.O;
    combine_prob.batch      = splitkv_prob.batch;
    combine_prob.nhead      = splitkv_prob.nhead;
    combine_prob.num_splits = splitkv_prob.num_splits;
    combine_prob.dtype      = ck::host::DataType::Half;

    const float scale_s = 1.0f / std::sqrt(static_cast<float>(splitkv_prob.K));

    constexpr int warmup_iters = 5;
    constexpr int bench_iters  = 100;

    auto splitkv_solutions = splitkv_prob.GetSolutions("gfx90a");
    auto combine_solutions = combine_prob.GetSolutions("gfx90a");

    std::cout << "\n=== Flash Decode Benchmark ===" << std::endl;
    std::cout << "Problem: batch=" << splitkv_prob.batch << ", nhead=" << splitkv_prob.nhead
              << ", nhead_k=" << splitkv_prob.nhead_k << ", M=" << splitkv_prob.M
              << ", N=" << splitkv_prob.N << ", K=" << splitkv_prob.K << ", O=" << splitkv_prob.O
              << ", num_splits=" << splitkv_prob.num_splits << std::endl;
    std::cout << "Warmup: " << warmup_iters << ", Iterations: " << bench_iters << std::endl;
    std::cout << "SplitKV solutions: " << splitkv_solutions.size() << std::endl;
    std::cout << "Combine solutions: " << combine_solutions.size() << std::endl;

    EXPECT(!splitkv_solutions.empty());
    EXPECT(!combine_solutions.empty());

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

    rtc::buffer<half> q_host(q_size), k_host(k_size), v_host(v_size);
    std::vector<float> q_ref(q_size), k_ref(k_size), v_ref(v_size), o_ref(o_size);

    auto fill_buffers = [&](auto& host, auto& ref) {
        for(std::size_t i = 0; i < host.size(); ++i)
        {
            float val = dist(rng);
            host[i]   = half(val);
            ref[i]    = val;
        }
    };
    fill_buffers(q_host, q_ref);
    fill_buffers(k_host, k_ref);
    fill_buffers(v_host, v_ref);

    ck::host::SplitKVParams splitkv_params = ck::host::make_splitkv_params(splitkv_prob, scale_s);
    ck::host::SplitKVCombineParams combine_params =
        ck::host::make_splitkv_combine_params(combine_prob);

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

    ck::host::device_fmha_fwd::cpu_attention_ref(q_ref, k_ref, v_ref, o_ref, ref_params);

    auto q_device = to_gpu(q_host);
    auto k_device = to_gpu(k_host);
    auto v_device = to_gpu(v_host);

    hipEvent_t splitkv_start, splitkv_stop, combine_start, combine_stop;
    (void)hipEventCreate(&splitkv_start);
    (void)hipEventCreate(&splitkv_stop);
    (void)hipEventCreate(&combine_start);
    (void)hipEventCreate(&combine_stop);

    ck::host::Solution& combine_sol = combine_solutions[0];
    std::cout << "\nCombine: " << combine_sol.ToTemplateString() << std::endl;

    auto combine_srcs = get_tile_headers_for_test();
    combine_srcs.push_back(
        {"main.cpp",
         make_splitkv_combine_kernel_source(combine_prob, combine_sol, combine_params)});
    rtc::compile_options combine_opts;
    combine_opts.kernel_name = "f";
    auto combine_kernel      = rtc::compile_kernel(combine_srcs, combine_opts);

    auto [combine_grid, combine_block] =
        ck::host::get_splitkv_combine_launch_dims(combine_sol, combine_prob);

    std::cout << "Combine Grid: (" << combine_grid.x << ", " << combine_grid.y << ", "
              << combine_grid.z << "), Block: (" << combine_block.x << ")" << std::endl;

    struct BenchResult
    {
        std::string splitkv_name;
        float splitkv_ms;
        float combine_ms;
        float total_ms;
        bool valid;
    };
    std::vector<BenchResult> results;

    for(std::size_t sol_idx = 0; sol_idx < splitkv_solutions.size(); ++sol_idx)
    {
        ck::host::Solution& splitkv_sol = splitkv_solutions[sol_idx];
        std::cout << "\n--- SplitKV solution " << (sol_idx + 1) << "/" << splitkv_solutions.size()
                  << " ---" << std::endl;
        std::cout << splitkv_sol.ToTemplateString() << std::endl;

        auto splitkv_srcs = get_tile_headers_for_test();
        splitkv_srcs.push_back(
            {"main.cpp", make_splitkv_kernel_source(splitkv_prob, splitkv_sol, splitkv_params)});
        rtc::compile_options splitkv_opts;
        splitkv_opts.kernel_name = "f";

        try
        {
            auto splitkv_kernel = rtc::compile_kernel(splitkv_srcs, splitkv_opts);

            auto [splitkv_grid, splitkv_block] =
                ck::host::get_splitkv_launch_dims(splitkv_sol, splitkv_prob);

            std::cout << "SplitKV Grid: (" << splitkv_grid.x << ", " << splitkv_grid.y << ", "
                      << splitkv_grid.z << "), Block: (" << splitkv_block.x << ")" << std::endl;

            rtc::buffer<float> o_acc_host(o_acc_size);
            rtc::buffer<float> lse_acc_host(lse_acc_size);
            std::fill(o_acc_host.begin(), o_acc_host.end(), 0.0f);
            std::fill(lse_acc_host.begin(),
                      lse_acc_host.end(),
                      -std::numeric_limits<float>::infinity());
            auto o_acc_device   = to_gpu(o_acc_host);
            auto lse_acc_device = to_gpu(lse_acc_host);

            rtc::buffer<half> o_host(o_size);
            std::fill(o_host.begin(), o_host.end(), half(0.0f));
            auto o_device = to_gpu(o_host);

            for(int i = 0; i < warmup_iters; ++i)
            {
                splitkv_kernel.launch(nullptr, splitkv_grid, splitkv_block)(q_device.data(),
                                                                            k_device.data(),
                                                                            v_device.data(),
                                                                            o_acc_device.data(),
                                                                            lse_acc_device.data());
                combine_kernel.launch(nullptr, combine_grid, combine_block)(
                    lse_acc_device.data(), o_acc_device.data(), o_device.data());
            }
            (void)hipDeviceSynchronize();

            o_host = rtc::from_gpu(o_device);
            std::vector<float> result(o_size);
            std::transform(o_host.begin(), o_host.end(), result.begin(), [](half v) {
                return static_cast<float>(v);
            });
            bool valid = allclose(o_ref, result, 0.0001, 0.0001);

            (void)hipEventRecord(splitkv_start, nullptr);
            for(int i = 0; i < bench_iters; ++i)
            {
                splitkv_kernel.launch(nullptr, splitkv_grid, splitkv_block)(q_device.data(),
                                                                            k_device.data(),
                                                                            v_device.data(),
                                                                            o_acc_device.data(),
                                                                            lse_acc_device.data());
            }
            (void)hipEventRecord(splitkv_stop, nullptr);
            (void)hipEventSynchronize(splitkv_stop);

            float splitkv_total_ms = 0.0f;
            (void)hipEventElapsedTime(&splitkv_total_ms, splitkv_start, splitkv_stop);
            float splitkv_avg_ms = splitkv_total_ms / bench_iters;

            (void)hipEventRecord(combine_start, nullptr);
            for(int i = 0; i < bench_iters; ++i)
            {
                combine_kernel.launch(nullptr, combine_grid, combine_block)(
                    lse_acc_device.data(), o_acc_device.data(), o_device.data());
            }
            (void)hipEventRecord(combine_stop, nullptr);
            (void)hipEventSynchronize(combine_stop);

            float combine_total_ms = 0.0f;
            (void)hipEventElapsedTime(&combine_total_ms, combine_start, combine_stop);
            float combine_avg_ms = combine_total_ms / bench_iters;

            float total_avg_ms = splitkv_avg_ms + combine_avg_ms;

            std::cout << "  SplitKV:  " << splitkv_avg_ms << " ms" << std::endl;
            std::cout << "  Combine:  " << combine_avg_ms << " ms" << std::endl;
            std::cout << "  Total:    " << total_avg_ms << " ms" << std::endl;
            std::cout << "  Valid:    " << (valid ? "yes" : "NO") << std::endl;

            results.push_back({splitkv_sol.ToTemplateString(),
                               splitkv_avg_ms,
                               combine_avg_ms,
                               total_avg_ms,
                               valid});

            CHECK(valid);
        }
        catch(const std::exception& e)
        {
            std::cout << "  COMPILE ERROR: " << e.what() << std::endl;
            results.push_back({splitkv_sol.ToTemplateString(), 0, 0, 0, false});
        }
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << std::left << std::setw(12) << "SplitKV(ms)" << std::setw(12) << "Combine(ms)"
              << std::setw(12) << "Total(ms)" << std::setw(8) << "Valid"
              << "Solution" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    float best_total     = std::numeric_limits<float>::max();
    std::size_t best_idx = 0;
    for(std::size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        std::cout << std::left << std::setw(12) << r.splitkv_ms << std::setw(12) << r.combine_ms
                  << std::setw(12) << r.total_ms << std::setw(8) << (r.valid ? "yes" : "NO")
                  << r.splitkv_name << std::endl;
        if(r.valid && r.total_ms < best_total)
        {
            best_total = r.total_ms;
            best_idx   = i;
        }
    }

    if(best_total < std::numeric_limits<float>::max())
    {
        std::cout << "\nBest: " << best_total << " ms - " << results[best_idx].splitkv_name
                  << std::endl;
    }

    (void)hipEventDestroy(splitkv_start);
    (void)hipEventDestroy(splitkv_stop);
    (void)hipEventDestroy(combine_start);
    (void)hipEventDestroy(combine_stop);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
