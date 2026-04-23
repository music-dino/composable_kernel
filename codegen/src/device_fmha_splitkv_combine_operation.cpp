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

    // OaccDataType is float, so MaxVectorSize = 16 / sizeof(float) = 4
    // NThreads = kN1 / MaxVectorSize = kN1 / 4
    // kM0 = warp_size / NThreads = 64 / (kN1 / 4) = 256 / kN1
    // TODO make architecture-specific
    constexpr std::size_t warp_size      = 64;
    constexpr std::size_t max_vector_size = 4; // 16 / sizeof(float)

    std::size_t log_max_splits = ComputeLogMaxSplits(prob.num_splits);
    std::size_t kMaxSplits     = 1ULL << log_max_splits;

    // MakeLSEaccRegTileDistribution caps NThreads at 8, so the real constraint is:
    // kM0 * min(kMaxSplits, 8) >= warp_size, and since kMaxSplits >= 8 always,
    // this simplifies to kM0 >= 8, i.e. kN1 <= 32
    constexpr std::size_t max_kN1 = (warp_size * max_vector_size) / 8; // 256 / 8 = 32

    for(int i = 3; i <= 8; ++i)
    {
        std::size_t kN1 = 1ULL << i;
        if(kN1 > prob.O || kN1 > max_kN1)
            break;
        if(prob.O & (kN1 - 1))
            continue;

        std::size_t kM0 = (warp_size * max_vector_size) / kN1;

        Operation op;
        op.hdim_v         = prob.O;
        op.n1             = kN1;
        op.log_max_splits = log_max_splits;
        op.dtype          = prob.dtype;
        op.pad_seqlen_q   = (prob.M % kM0 != 0);
        op.pad_hdim_v     = false; // kN1 divides hdim_v exactly
        result.push_back(op);
    }

    return result;
}

using device_fmha_common::ToDataTypeString;

Solution Operation::ToSolution() const
{
    constexpr std::size_t warp_size       = 64;
    constexpr std::size_t max_vector_size_ = 4; // 16 / sizeof(float)
    std::size_t m0 = (warp_size * max_vector_size_) / n1;

    std::unordered_map<std::string, std::string> values = {
        {"DataType", ToDataTypeString(dtype)},
        {"HeadDimV", std::to_string(hdim_v)},
        {"N1", std::to_string(n1)},
        {"M0", std::to_string(m0)},
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
