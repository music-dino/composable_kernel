// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_splitkv {

struct Problem
{
    std::size_t M = 0; // seqlen_q (typically 1 for decode)
    std::size_t N = 0; // seqlen_k (full KV cache length)
    std::size_t K = 0; // hdim_q
    std::size_t O = 0; // hdim_v

    std::size_t batch      = 0;
    std::size_t nhead      = 0; // nhead_q (number of Q heads)
    std::size_t nhead_k    = 0; // nhead_k (number of K/V heads, nhead_k <= nhead)
    std::size_t num_splits = 0; // number of KV splits for flash decoding

    DataType dtype = DataType::Half;

    bool is_v_rowmajor = true; // true=[N,O], false=[O,N]

    std::string GetIncludeHeader() const;

    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_splitkv
} // namespace host
} // namespace ck
