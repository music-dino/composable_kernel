// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/host/types.hpp"
#include <string>
#include <vector>

namespace ck {
namespace host {
namespace device_fmha_appendkv {

enum class RotaryEmbeddingType
{
    None,
    Interleaved,
    HalfRotated
};

struct Problem
{
    std::size_t M = 0; // seqlen_q
    std::size_t N = 0; // seqlen_knew (new KV to append)
    std::size_t K = 0; // hdim_q
    std::size_t O = 0; // hdim_v

    std::size_t batch   = 0;
    std::size_t nhead   = 0;
    std::size_t nhead_k = 0;

    std::size_t cache_seqlen   = 0; // Current cache length (append position)
    std::size_t total_seqlen_k = 0; // Total KV cache capacity
    std::size_t rotary_dim     = 0; // Rotary embedding dimension (0 = no RoPE, typically = K)

    DataType dtype             = DataType::Half;
    bool is_v_rowmajor         = true;
    bool has_mask              = false;
    RotaryEmbeddingType rotary = RotaryEmbeddingType::None;

    std::string GetIncludeHeader() const;
    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_appendkv
} // namespace host
} // namespace ck
