// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/device_fmha_splitkv_combine/problem.hpp"

namespace ck {
namespace host {
namespace device_fmha_splitkv_combine {

struct Operation
{
    std::size_t hdim_v         = 0;  // kHeadDimV
    std::size_t n1             = 32; // kN1 (tile size for hdim_v, always 32 for fp16)
    std::size_t log_max_splits = 3;  // kLogMaxSplits (log2 of rounded num_splits, min 3)

    DataType dtype = DataType::Half;

    bool pad_seqlen_q = true; // kPadSeqLenQ
    bool pad_hdim_v   = true; // kPadHeadDimV

    static std::vector<Operation> CreateOperations(const Problem& prob, const std::string& arch);

    Solution ToSolution() const;
};

bool IsSupportedArch(const std::string& arch);

} // namespace device_fmha_splitkv_combine
} // namespace host
} // namespace ck
