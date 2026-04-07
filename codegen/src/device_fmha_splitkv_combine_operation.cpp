// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_splitkv_combine/operation.hpp"
#include "ck/host/device_fmha_splitkv_combine/problem.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/device_fmha_common/tile_config.hpp"
#include <string>
#include <vector>
#include <algorithm>

namespace ck {
namespace host {
namespace device_fmha_splitkv_combine {

static const char* const FmhaFwdSplitKVCombineWrapperTemplate =
    "ck_tile::FmhaFwdSplitKVCombineWrapper<${DataType}, "
    "${HeadDimV}, ${N1}, ${LogMaxSplits}, "
    "${PadSeqLenQ}, ${PadHeadDimV}>";

static bool IsGfx9(const std::string& arch)
{
    return arch.find("gfx9") == 0 && arch.find("gfx950") != 0;
}

static bool IsGfx950(const std::string& arch) { return arch.find("gfx950") == 0; }

bool IsSupportedArch(const std::string& arch) { return IsGfx9(arch) || IsGfx950(arch); }

// Compute kLogMaxSplits: log2 of the smallest power-of-2 >= num_splits, with minimum of 8
// Returns value in range [3, 7] corresponding to max splits of [8, 16, 32, 64, 128]
static std::size_t ComputeLogMaxSplits(std::size_t num_splits)
{
    std::size_t log_val = 3; // minimum is 2^3 = 8
    while((1ULL << log_val) < num_splits && log_val < 7)
        ++log_val;
    return log_val;
}

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    std::vector<Operation> result;

    if(!IsSupportedArch(arch))
        return result;

    if(prob.dtype != DataType::Half)
        return result;

    // Constants for Combine kernel
    constexpr std::size_t kN1 = 32; // tile size for hdim_v (fp16)
    constexpr std::size_t kM0 = 8;  // tile size for seqlen_q (derived from kN1)

    // Compute exact padding needs
    bool needs_pad_seqlen_q = (prob.M % kM0 != 0);
    bool needs_pad_hdim_v   = (prob.O % kN1 != 0);

    // Compute log_max_splits
    std::size_t log_max_splits = ComputeLogMaxSplits(prob.num_splits);

    Operation op;
    op.hdim_v         = prob.O;
    op.n1             = kN1;
    op.log_max_splits = log_max_splits;
    op.dtype          = prob.dtype;
    op.pad_seqlen_q   = needs_pad_seqlen_q;
    op.pad_hdim_v     = needs_pad_hdim_v;

    result.push_back(op);

    return result;
}

using device_fmha_common::ToDataTypeString;

Solution Operation::ToSolution() const
{
    std::unordered_map<std::string, std::string> values = {
        {"DataType", ToDataTypeString(dtype)},
        {"HeadDimV", std::to_string(hdim_v)},
        {"N1", std::to_string(n1)},
        {"LogMaxSplits", std::to_string(log_max_splits)},
        {"PadSeqLenQ", pad_seqlen_q ? "true" : "false"},
        {"PadHeadDimV", pad_hdim_v ? "true" : "false"},
    };

    return Solution{InterpolateString(FmhaFwdSplitKVCombineWrapperTemplate, values),
                    std::move(values)};
}

} // namespace device_fmha_splitkv_combine
} // namespace host
} // namespace ck
