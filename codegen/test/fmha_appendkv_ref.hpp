// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/host/device_fmha_appendkv/problem.hpp"
#include <cstdint>
#include <vector>
#include <algorithm>

namespace ck {
namespace host {

using RotaryEmbeddingType = device_fmha_appendkv::RotaryEmbeddingType;

struct AppendKVRefParams
{
    std::size_t batch;
    std::size_t nhead;
    std::size_t nhead_k;
    std::size_t seqlen_q;
    std::size_t seqlen_knew;
    std::size_t cache_seqlen;
    std::size_t hdim_q;
    std::size_t hdim_v;

    // Q strides [batch, nhead, seqlen_q, hdim_q]
    std::size_t q_stride_batch;
    std::size_t q_stride_nhead;
    std::size_t q_stride_seq;

    // K cache strides [batch, nhead_k, total_seqlen_k, hdim_q]
    std::size_t k_stride_batch;
    std::size_t k_stride_nhead;
    std::size_t k_stride_seq;

    // Knew strides [batch, nhead_k, seqlen_knew, hdim_q]
    std::size_t knew_stride_batch;
    std::size_t knew_stride_nhead;
    std::size_t knew_stride_seq;

    // V cache strides [batch, nhead_k, total_seqlen_k, hdim_v]
    std::size_t v_stride_batch;
    std::size_t v_stride_nhead;
    std::size_t v_stride_seq;

    // Vnew strides [batch, nhead_k, seqlen_knew, hdim_v]
    std::size_t vnew_stride_batch;
    std::size_t vnew_stride_nhead;
    std::size_t vnew_stride_seq;
};

// CPU reference for AppendKV with optional RoPE
// - Applies RoPE to Q and Knew (if cos/sin provided)
// - Appends (rotated) Knew to K cache at position cache_seqlen
// - Appends Vnew to V cache at position cache_seqlen
//
// Rotary embedding types (both use cos/sin shape [total_seqlen, rotary_dim / 2]):
// - HalfRotated:  pairs (d, d+half_dim)
// - Interleaved:  pairs (2d, 2d+1)
inline void cpu_appendkv_rope_ref(std::vector<float>& q,
                                  const std::vector<float>& knew,
                                  const std::vector<float>& vnew,
                                  std::vector<float>& k_cache,
                                  std::vector<float>& v_cache,
                                  const std::vector<float>& cos,
                                  const std::vector<float>& sin,
                                  const AppendKVRefParams& p,
                                  RotaryEmbeddingType rotary_type = RotaryEmbeddingType::None)
{
    const bool apply_rope =
        rotary_type != RotaryEmbeddingType::None && !cos.empty() && !sin.empty();
    const std::size_t half_dim = p.hdim_q / 2;

    // Apply RoPE to Q
    if(apply_rope)
    {
        for(std::size_t b = 0; b < p.batch; ++b)
        {
            for(std::size_t h = 0; h < p.nhead; ++h)
            {
                float* q_ptr = q.data() + b * p.q_stride_batch + h * p.q_stride_nhead;

                for(std::size_t s = 0; s < p.seqlen_q; ++s)
                {
                    std::size_t pos      = p.cache_seqlen + s;
                    const float* cos_ptr = cos.data() + pos * half_dim;
                    const float* sin_ptr = sin.data() + pos * half_dim;

                    for(std::size_t d = 0; d < half_dim; ++d)
                    {
                        float x0, x1, c, sn;
                        std::size_t idx0, idx1;

                        if(rotary_type == RotaryEmbeddingType::Interleaved)
                        {
                            idx0 = 2 * d;
                            idx1 = 2 * d + 1;
                            c    = cos_ptr[d];
                            sn   = sin_ptr[d];
                        }
                        else // HalfRotated
                        {
                            idx0 = d;
                            idx1 = d + half_dim;
                            c    = cos_ptr[d];
                            sn   = sin_ptr[d];
                        }

                        x0                               = q_ptr[s * p.q_stride_seq + idx0];
                        x1                               = q_ptr[s * p.q_stride_seq + idx1];
                        q_ptr[s * p.q_stride_seq + idx0] = x0 * c - x1 * sn;
                        q_ptr[s * p.q_stride_seq + idx1] = x1 * c + x0 * sn;
                    }
                }
            }
        }
    }

    // Apply RoPE to Knew and append to K cache
    for(std::size_t b = 0; b < p.batch; ++b)
    {
        for(std::size_t h = 0; h < p.nhead_k; ++h)
        {
            const float* knew_ptr = knew.data() + b * p.knew_stride_batch + h * p.knew_stride_nhead;
            float* k_ptr          = k_cache.data() + b * p.k_stride_batch + h * p.k_stride_nhead;

            for(std::size_t s = 0; s < p.seqlen_knew; ++s)
            {
                std::size_t pos = p.cache_seqlen + s;

                if(apply_rope)
                {
                    const float* cos_ptr = cos.data() + pos * half_dim;
                    const float* sin_ptr = sin.data() + pos * half_dim;

                    for(std::size_t d = 0; d < half_dim; ++d)
                    {
                        float x0, x1, c, sn;
                        std::size_t idx0, idx1;

                        if(rotary_type == RotaryEmbeddingType::Interleaved)
                        {
                            idx0 = 2 * d;
                            idx1 = 2 * d + 1;
                            c    = cos_ptr[d];
                            sn   = sin_ptr[d];
                        }
                        else // HalfRotated
                        {
                            idx0 = d;
                            idx1 = d + half_dim;
                            c    = cos_ptr[d];
                            sn   = sin_ptr[d];
                        }

                        x0                                 = knew_ptr[s * p.knew_stride_seq + idx0];
                        x1                                 = knew_ptr[s * p.knew_stride_seq + idx1];
                        k_ptr[pos * p.k_stride_seq + idx0] = x0 * c - x1 * sn;
                        k_ptr[pos * p.k_stride_seq + idx1] = x1 * c + x0 * sn;
                    }
                }
                else
                {
                    for(std::size_t d = 0; d < p.hdim_q; ++d)
                    {
                        k_ptr[pos * p.k_stride_seq + d] = knew_ptr[s * p.knew_stride_seq + d];
                    }
                }
            }
        }
    }

    // Append Vnew to V cache (no RoPE for V)
    for(std::size_t b = 0; b < p.batch; ++b)
    {
        for(std::size_t h = 0; h < p.nhead_k; ++h)
        {
            const float* vnew_ptr = vnew.data() + b * p.vnew_stride_batch + h * p.vnew_stride_nhead;
            float* v_ptr          = v_cache.data() + b * p.v_stride_batch + h * p.v_stride_nhead;

            for(std::size_t s = 0; s < p.seqlen_knew; ++s)
            {
                std::size_t pos = p.cache_seqlen + s;
                for(std::size_t d = 0; d < p.hdim_v; ++d)
                {
                    v_ptr[pos * p.v_stride_seq + d] = vnew_ptr[s * p.vnew_stride_seq + d];
                }
            }
        }
    }
}

} // namespace host
} // namespace ck
