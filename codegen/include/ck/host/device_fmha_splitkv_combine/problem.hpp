// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <string>
#include <vector>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_splitkv_combine {

struct Problem
{
    std::size_t M = 0; // seqlen_q
    std::size_t O = 0; // hdim_v

    std::size_t batch      = 0;
    std::size_t nhead      = 0; // nhead_q
    std::size_t num_splits = 0; // number of KV splits

    DataType dtype = DataType::Half;

    std::string GetIncludeHeader() const;
    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_splitkv_combine
} // namespace host
} // namespace ck
