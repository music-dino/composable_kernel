// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host/device_fmha_appendkv/problem.hpp"
#include "ck/host/device_fmha_appendkv/operation.hpp"
#include "fmha_appendkv_ref.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "ck/host/headers.hpp"
#include "common.hpp"
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include <test.hpp>
#include <string>

namespace appendkv = ck::host::device_fmha_appendkv;

static const std::string appendkv_kernel_template = R"__ck__(
#include <cmath>
#include <cstdint>
#include <cassert>
#include <${include}>

using KernelType = ${template};

extern "C" __launch_bounds__(KernelType::Kernel::kBlockSize, KernelType::Kernel::kBlockPerCu)
__global__ void f(${dtype}* q,
                  ${dtype}* k,
                  const ${dtype}* knew,
                  ${dtype}* v,
                  const ${dtype}* vnew,
                  const int32_t* seqlen_k_ptr,
                  const ${dtype}* rotary_cos,
                  const ${dtype}* rotary_sin) {
    
    using Kernel = KernelType;
    
    constexpr auto desc = Kernel::make_descriptor(
        ck_tile::make_tuple(${batch}, ${nhead}, ${seqlen_q}, ${hdim_q}),
        ck_tile::make_tuple(${q_stride_batch}, ${q_stride_nhead}, ${q_stride_seqlen}),
        ck_tile::make_tuple(${batch}, ${nhead_k}, ${seqlen_k}, ${hdim_q}),
        ck_tile::make_tuple(${k_stride_batch}, ${k_stride_nhead}, ${k_stride_seqlen}),
        ck_tile::make_tuple(${batch}, ${nhead_k}, ${seqlen_knew}, ${hdim_q}),
        ck_tile::make_tuple(${knew_stride_batch}, ${knew_stride_nhead}, ${knew_stride_seqlen}),
        ck_tile::make_tuple(${batch}, ${nhead_k}, ${seqlen_k}, ${hdim_v}),
        ck_tile::make_tuple(${v_stride_batch}, ${v_stride_nhead}, ${v_stride_seqlen}),
        ck_tile::make_tuple(${batch}, ${nhead_k}, ${seqlen_knew}, ${hdim_v}),
        ck_tile::make_tuple(${vnew_stride_batch}, ${vnew_stride_nhead}, ${vnew_stride_seqlen}),
        ${rotary_dim},
        ${has_mask});
    
    static_assert(desc.IsValid(), "Invalid AppendKV kernel configuration");
    
    Kernel::Run(desc, q, k, knew, v, vnew, seqlen_k_ptr, rotary_cos, rotary_sin, nullptr);
}
)__ck__";

inline std::string make_appendkv_kernel_source(const appendkv::Problem& prob,
                                               const ck::host::Solution& solution,
                                               const ck::host::AppendKVRefParams& p)
{
    return ck::host::InterpolateString(appendkv_kernel_template,
                                       {{"include", prob.GetIncludeHeader()},
                                        {"template", solution.ToTemplateString()},
                                        {"dtype", "ck_tile::fp16_t"},
                                        {"batch", std::to_string(p.batch)},
                                        {"nhead", std::to_string(p.nhead)},
                                        {"nhead_k", std::to_string(p.nhead_k)},
                                        {"seqlen_q", std::to_string(p.seqlen_q)},
                                        {"seqlen_k", std::to_string(prob.total_seqlen_k)},
                                        {"seqlen_knew", std::to_string(p.seqlen_knew)},
                                        {"hdim_q", std::to_string(p.hdim_q)},
                                        {"hdim_v", std::to_string(p.hdim_v)},
                                        {"rotary_dim", std::to_string(prob.rotary_dim)},
                                        {"has_mask", prob.has_mask ? "true" : "false"},
                                        {"q_stride_batch", std::to_string(p.q_stride_batch)},
                                        {"q_stride_nhead", std::to_string(p.q_stride_nhead)},
                                        {"q_stride_seqlen", std::to_string(p.q_stride_seq)},
                                        {"k_stride_batch", std::to_string(p.k_stride_batch)},
                                        {"k_stride_nhead", std::to_string(p.k_stride_nhead)},
                                        {"k_stride_seqlen", std::to_string(p.k_stride_seq)},
                                        {"knew_stride_batch", std::to_string(p.knew_stride_batch)},
                                        {"knew_stride_nhead", std::to_string(p.knew_stride_nhead)},
                                        {"knew_stride_seqlen", std::to_string(p.knew_stride_seq)},
                                        {"v_stride_batch", std::to_string(p.v_stride_batch)},
                                        {"v_stride_nhead", std::to_string(p.v_stride_nhead)},
                                        {"v_stride_seqlen", std::to_string(p.v_stride_seq)},
                                        {"vnew_stride_batch", std::to_string(p.vnew_stride_batch)},
                                        {"vnew_stride_nhead", std::to_string(p.vnew_stride_nhead)},
                                        {"vnew_stride_seqlen", std::to_string(p.vnew_stride_seq)}});
}

static ck::host::AppendKVRefParams make_appendkv_ref_params(const appendkv::Problem& prob)
{
    ck::host::AppendKVRefParams ref_params;
    ref_params.batch        = prob.batch;
    ref_params.nhead        = prob.nhead;
    ref_params.nhead_k      = prob.nhead_k;
    ref_params.seqlen_q     = prob.M;
    ref_params.seqlen_knew  = prob.N;
    ref_params.cache_seqlen = prob.cache_seqlen;
    ref_params.hdim_q       = prob.K;
    ref_params.hdim_v       = prob.O;
    // Q strides [batch, nhead, seqlen_q, hdim_q]
    ref_params.q_stride_seq   = prob.K;
    ref_params.q_stride_nhead = prob.M * prob.K;
    ref_params.q_stride_batch = prob.nhead * prob.M * prob.K;
    // K cache strides [batch, nhead_k, total_seqlen_k, hdim_q]
    ref_params.k_stride_seq   = prob.K;
    ref_params.k_stride_nhead = prob.total_seqlen_k * prob.K;
    ref_params.k_stride_batch = prob.nhead_k * prob.total_seqlen_k * prob.K;
    // Knew strides [batch, nhead_k, seqlen_knew, hdim_q]
    ref_params.knew_stride_seq   = prob.K;
    ref_params.knew_stride_nhead = prob.N * prob.K;
    ref_params.knew_stride_batch = prob.nhead_k * prob.N * prob.K;
    // V cache strides [batch, nhead_k, total_seqlen_k, hdim_v]
    ref_params.v_stride_seq   = prob.O;
    ref_params.v_stride_nhead = prob.total_seqlen_k * prob.O;
    ref_params.v_stride_batch = prob.nhead_k * prob.total_seqlen_k * prob.O;
    // Vnew strides [batch, nhead_k, seqlen_knew, hdim_v]
    ref_params.vnew_stride_seq   = prob.O;
    ref_params.vnew_stride_nhead = prob.N * prob.O;
    ref_params.vnew_stride_batch = prob.nhead_k * prob.N * prob.O;
    return ref_params;
}

inline std::pair<dim3, dim3> get_appendkv_launch_dims(const ck::host::Solution& solution,
                                                      const appendkv::Problem& prob)
{
    // From fmha_fwd_appendkv_kernel.hpp:
    // grid.x = max(ceil(seqlen_q / kM0), ceil(seqlen_knew / kN0))
    // grid.y = nhead
    // grid.z = batch
    // BlockSize = 256 (hardcoded in BlockFmhaFwdAppendKVPipelineProblem)

    auto m0 = solution.GetTemplateParameter<std::size_t>("M0");
    auto n0 = solution.GetTemplateParameter<std::size_t>("N0");

    std::size_t num_tiles_m = ck::host::integer_divide_ceil(prob.M, m0);
    std::size_t num_tiles_n = ck::host::integer_divide_ceil(prob.N, n0);
    std::size_t num_tiles   = std::max(num_tiles_m, num_tiles_n);

    constexpr std::size_t block_size = 256;

    dim3 grid(num_tiles, prob.nhead, prob.batch);
    dim3 block(block_size);

    return {grid, block};
}

TEST_CASE(test_appendkv_basic)
{
    appendkv::Problem prob;
    prob.M              = 1;  // seqlen_q
    prob.N              = 8;  // seqlen_knew (new KV to append)
    prob.K              = 64; // hdim_q
    prob.O              = 64; // hdim_v
    prob.batch          = 2;
    prob.nhead          = 4;
    prob.nhead_k        = 4;
    prob.dtype          = ck::host::DataType::Half;
    prob.is_v_rowmajor  = true;
    prob.rotary         = appendkv::RotaryEmbeddingType::None;
    prob.cache_seqlen   = 64;
    prob.total_seqlen_k = prob.cache_seqlen + prob.N;

    auto solutions = prob.GetSolutions("gfx90a");
    std::cout << "Number of AppendKV solutions: " << solutions.size() << std::endl;

    EXPECT(!solutions.empty());

    // Generate random input data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const std::size_t q_size    = prob.batch * prob.nhead * prob.M * prob.K;
    const std::size_t k_size    = prob.batch * prob.nhead_k * prob.total_seqlen_k * prob.K;
    const std::size_t knew_size = prob.batch * prob.nhead_k * prob.N * prob.K;
    const std::size_t v_size    = prob.batch * prob.nhead_k * prob.total_seqlen_k * prob.O;
    const std::size_t vnew_size = prob.batch * prob.nhead_k * prob.N * prob.O;

    std::vector<float> q_data(q_size);
    std::vector<float> k_data(k_size);
    std::vector<float> knew_data(knew_size);
    std::vector<float> v_data(v_size);
    std::vector<float> vnew_data(vnew_size);
    std::generate(q_data.begin(), q_data.end(), [&]() { return dist(rng); });
    std::generate(k_data.begin(), k_data.end(), [&]() { return dist(rng); });
    std::generate(knew_data.begin(), knew_data.end(), [&]() { return dist(rng); });
    std::generate(v_data.begin(), v_data.end(), [&]() { return dist(rng); });
    std::generate(vnew_data.begin(), vnew_data.end(), [&]() { return dist(rng); });

    // seqlen_k per batch (where to append)
    std::vector<int32_t> seqlen_k_data(prob.batch, static_cast<int32_t>(prob.cache_seqlen));

    // Compute reference (no RoPE - empty cos/sin)
    auto ref_params          = make_appendkv_ref_params(prob);
    std::vector<float> q_ref = q_data;
    std::vector<float> k_ref = k_data;
    std::vector<float> v_ref = v_data;
    cpu_appendkv_rope_ref(q_ref, knew_data, vnew_data, k_ref, v_ref, {}, {}, ref_params);

    const auto make_device_buff = [](const std::vector<float>& data) {
        rtc::buffer<_Float16> host(data.size());
        std::transform(
            data.begin(), data.end(), host.begin(), [](float val) { return _Float16(val); });
        return to_gpu(host);
    };

    auto q_device    = make_device_buff(q_data);
    auto k_device    = make_device_buff(k_data);
    auto knew_device = make_device_buff(knew_data);
    auto v_device    = make_device_buff(v_data);
    auto vnew_device = make_device_buff(vnew_data);
    rtc::buffer<int32_t> seqlen_k_host(seqlen_k_data.size());
    std::copy(seqlen_k_data.begin(), seqlen_k_data.end(), seqlen_k_host.begin());
    auto seqlen_k_device = to_gpu(seqlen_k_host);

    for(std::size_t sol_idx = 0; sol_idx < solutions.size(); ++sol_idx)
    {
        auto&& solution = solutions[sol_idx];
        std::cout << "Testing solution " << (sol_idx + 1) << "/" << solutions.size() << ": "
                  << solution.ToTemplateString() << std::endl;

        auto srcs = get_tile_headers_for_test();
        srcs.push_back({"main.cpp", make_appendkv_kernel_source(prob, solution, ref_params)});

        rtc::compile_options options;
        options.kernel_name = "f";
        auto kernel         = rtc::compile_kernel(srcs, options);

        auto [grid, block] = get_appendkv_launch_dims(solution, prob);

        std::cout << "  Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << "), "
                  << "Block: (" << block.x << ")" << std::endl;

        // Reset K and V buffers before running kernel
        k_device = make_device_buff(k_data);
        v_device = make_device_buff(v_data);

        kernel.launch(nullptr, grid, block)(q_device.data(),
                                            k_device.data(),
                                            knew_device.data(),
                                            v_device.data(),
                                            vnew_device.data(),
                                            seqlen_k_device.data(),
                                            static_cast<_Float16*>(nullptr),  // rotary_cos
                                            static_cast<_Float16*>(nullptr)); // rotary_sin

        // Copy back and verify
        auto k_host = rtc::from_gpu(k_device);
        auto v_host = rtc::from_gpu(v_device);

        // Convert to float for comparison
        std::vector<float> k_result(k_size);
        std::vector<float> v_result(v_size);
        std::transform(k_host.begin(), k_host.end(), k_result.begin(), [](auto val) {
            return static_cast<float>(val);
        });
        std::transform(v_host.begin(), v_host.end(), v_result.begin(), [](auto val) {
            return static_cast<float>(val);
        });

        CHECK(allclose(k_result, k_ref, 0.001, 0.001));
        CHECK(allclose(v_result, v_ref, 0.001, 0.001));
        std::cout << "  PASSED" << std::endl;
    }
}

static void test_appendkv_rope_impl(appendkv::RotaryEmbeddingType rotary_type)
{
    std::string type_name =
        (rotary_type == appendkv::RotaryEmbeddingType::HalfRotated) ? "HalfRotated" : "Interleaved";

    appendkv::Problem prob;
    prob.M              = 1;  // seqlen_q
    prob.N              = 8;  // seqlen_knew (new KV to append)
    prob.K              = 64; // hdim_q
    prob.O              = 64; // hdim_v
    prob.batch          = 2;
    prob.nhead          = 4;
    prob.nhead_k        = 4;
    prob.dtype          = ck::host::DataType::Half;
    prob.is_v_rowmajor  = true;
    prob.rotary         = rotary_type;
    prob.cache_seqlen   = 64;
    prob.total_seqlen_k = prob.cache_seqlen + prob.N;
    prob.rotary_dim     = prob.K; // Full hdim rotation

    auto solutions = prob.GetSolutions("gfx90a");
    std::cout << "Number of AppendKV (" << type_name << ") solutions: " << solutions.size()
              << std::endl;

    EXPECT(!solutions.empty());

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> cos_sin_dist(-1.0f, 1.0f);

    const std::size_t q_size    = prob.batch * prob.nhead * prob.M * prob.K;
    const std::size_t k_size    = prob.batch * prob.nhead_k * prob.total_seqlen_k * prob.K;
    const std::size_t knew_size = prob.batch * prob.nhead_k * prob.N * prob.K;
    const std::size_t v_size    = prob.batch * prob.nhead_k * prob.total_seqlen_k * prob.O;
    const std::size_t vnew_size = prob.batch * prob.nhead_k * prob.N * prob.O;

    std::vector<float> q_data(q_size);
    std::vector<float> k_data(k_size);
    std::vector<float> knew_data(knew_size);
    std::vector<float> v_data(v_size);
    std::vector<float> vnew_data(vnew_size);
    std::generate(q_data.begin(), q_data.end(), [&]() { return dist(rng); });
    std::generate(k_data.begin(), k_data.end(), [&]() { return dist(rng); });
    std::generate(knew_data.begin(), knew_data.end(), [&]() { return dist(rng); });
    std::generate(v_data.begin(), v_data.end(), [&]() { return dist(rng); });
    std::generate(vnew_data.begin(), vnew_data.end(), [&]() { return dist(rng); });

    std::vector<int32_t> seqlen_k_data(prob.batch, static_cast<int32_t>(prob.cache_seqlen));

    const std::size_t half_dim = prob.rotary_dim / 2;

    // cos/sin table: [total_seqlen_k, half_dim] - same shape for both embedding types
    std::vector<float> cos_data(prob.total_seqlen_k * half_dim);
    std::vector<float> sin_data(prob.total_seqlen_k * half_dim);
    std::generate(cos_data.begin(), cos_data.end(), [&]() { return cos_sin_dist(rng); });
    std::generate(sin_data.begin(), sin_data.end(), [&]() { return cos_sin_dist(rng); });

    // Compute reference
    auto ref_params          = make_appendkv_ref_params(prob);
    std::vector<float> q_ref = q_data;
    std::vector<float> k_ref = k_data;
    std::vector<float> v_ref = v_data;
    cpu_appendkv_rope_ref(
        q_ref, knew_data, vnew_data, k_ref, v_ref, cos_data, sin_data, ref_params, rotary_type);

    const auto make_device_buff = [](const std::vector<float>& data) {
        rtc::buffer<_Float16> host(data.size());
        std::transform(
            data.begin(), data.end(), host.begin(), [](float val) { return _Float16(val); });
        return to_gpu(host);
    };

    auto q_device    = make_device_buff(q_data);
    auto k_device    = make_device_buff(k_data);
    auto knew_device = make_device_buff(knew_data);
    auto v_device    = make_device_buff(v_data);
    auto vnew_device = make_device_buff(vnew_data);
    rtc::buffer<int32_t> seqlen_k_host(seqlen_k_data.size());
    std::copy(seqlen_k_data.begin(), seqlen_k_data.end(), seqlen_k_host.begin());
    auto seqlen_k_device = to_gpu(seqlen_k_host);
    auto cos_device      = make_device_buff(cos_data);
    auto sin_device      = make_device_buff(sin_data);

    for(std::size_t sol_idx = 0; sol_idx < solutions.size(); ++sol_idx)
    {
        auto&& solution = solutions[sol_idx];
        std::cout << "Testing " << type_name << " solution " << (sol_idx + 1) << "/"
                  << solutions.size() << ": " << solution.ToTemplateString() << std::endl;

        auto srcs = get_tile_headers_for_test();
        srcs.push_back({"main.cpp", make_appendkv_kernel_source(prob, solution, ref_params)});

        rtc::compile_options options;
        options.kernel_name = "f";
        auto kernel         = rtc::compile_kernel(srcs, options);

        auto [grid, block] = get_appendkv_launch_dims(solution, prob);

        std::cout << "  Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << "), "
                  << "Block: (" << block.x << ")" << std::endl;

        // Reset buffers
        q_device = make_device_buff(q_data);
        k_device = make_device_buff(k_data);
        v_device = make_device_buff(v_data);

        kernel.launch(nullptr, grid, block)(q_device.data(),
                                            k_device.data(),
                                            knew_device.data(),
                                            v_device.data(),
                                            vnew_device.data(),
                                            seqlen_k_device.data(),
                                            cos_device.data(),
                                            sin_device.data());

        auto q_host = rtc::from_gpu(q_device);
        auto k_host = rtc::from_gpu(k_device);
        auto v_host = rtc::from_gpu(v_device);

        std::vector<float> q_result(q_size);
        std::vector<float> k_result(k_size);
        std::vector<float> v_result(v_size);
        std::transform(q_host.begin(), q_host.end(), q_result.begin(), [](auto val) {
            return static_cast<float>(val);
        });
        std::transform(k_host.begin(), k_host.end(), k_result.begin(), [](auto val) {
            return static_cast<float>(val);
        });
        std::transform(v_host.begin(), v_host.end(), v_result.begin(), [](auto val) {
            return static_cast<float>(val);
        });

        CHECK(allclose(q_result, q_ref, 0.001, 0.001));
        CHECK(allclose(k_result, k_ref, 0.001, 0.001));
        CHECK(allclose(v_result, v_ref, 0.001, 0.001));
        std::cout << "  PASSED" << std::endl;
    }
}

TEST_CASE(test_appendkv_rope_half_rotated)
{
    test_appendkv_rope_impl(appendkv::RotaryEmbeddingType::HalfRotated);
}

TEST_CASE(test_appendkv_rope_interleaved)
{
    test_appendkv_rope_impl(appendkv::RotaryEmbeddingType::Interleaved);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
