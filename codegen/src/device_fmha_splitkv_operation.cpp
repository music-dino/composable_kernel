// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_splitkv/operation.hpp"
#include "ck/host/device_fmha_splitkv/problem.hpp"
#include "ck/host/stringutils.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace ck {
namespace host {
namespace device_fmha_splitkv {

static const char* const FmhaFwdSplitKVWrapperTemplate =
    "ck_tile::FmhaFwdSplitKVWrapper<${DataType}, ${OaccOutType}, "
    "${BM0}, ${BN0}, ${BK0}, ${BN1}, ${BK1}, ${BK0Max}, "
    "${RM0}, ${RN0}, ${RK0}, ${RM1}, ${RN1}, ${RK1}, "
    "${WM0}, ${WN0}, ${WK0}, ${WM1}, ${WN1}, ${WK1}, "
    "${IsVRowMajor}, "
    "${PadM}, ${PadN}, ${PadK}, ${PadO}, "
    "${HasUnevenSplits}, ${MergeNumHeadGroupsSeqLenQ}, "
    "ck_tile::FmhaSplitKVPipelineTag::${PipelineTag}>";

using device_fmha_common::IsGfx9;
using device_fmha_common::IsGfx950;
using device_fmha_common::CeilToValidHdim;
using device_fmha_common::IsValidVLdsConfig;

bool IsSupportedArch(const std::string& arch)
{
    return device_fmha_common::IsSupportedFmhaArch(arch);
}

// Base shape: tile parameters independent of K/O head dimensions.
// bn1 and bk0max are stamped in at generation time from the problem dimensions.
struct BaseShape
{
    std::size_t bm0, bn0, bk0;
    std::size_t rm0, rn0, rk0;
    std::size_t rm1, rn1, rk1;
    std::size_t wm0, wn0, wk0;
    std::size_t wm1, wn1, wk1;
};

// clang-format off
static const std::vector<BaseShape> gfx9_base_shapes = {
    // MFMA 16x16x16, bk0=16 — for small bk0max (32, 48, 80)
    { 16,  32, 16,  1, 1, 1,  1, 1, 1,  16, 16, 16,  16, 16, 16},
    { 16,  64, 16,  1, 1, 1,  1, 1, 1,  16, 16, 16,  16, 16, 16},
    { 32,  32, 16,  2, 1, 1,  2, 1, 1,  16, 16, 16,  16, 16, 16},
    { 32,  64, 16,  2, 1, 1,  2, 1, 1,  16, 16, 16,  16, 16, 16},
    { 64,  32, 16,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16},
    { 64,  64, 16,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16},
    {128,  64, 16,  8, 1, 1,  8, 1, 1,  16, 16, 16,  16, 16, 16},
    // MFMA 32x32x16, bk0=16
    { 32, 128, 16,  1, 1, 1,  1, 1, 1,  32, 32, 16,  32, 32, 16},
    { 32,  64, 16,  1, 1, 1,  1, 1, 1,  32, 32, 16,  32, 32, 16},
    { 64,  64, 16,  2, 1, 1,  2, 1, 1,  32, 32, 16,  32, 32, 16},
    { 64, 128, 16,  2, 1, 1,  2, 1, 1,  32, 32, 16,  32, 32, 16},
    {128,  64, 16,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16},
    {128, 128, 16,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16},
    { 32, 128, 16,  2, 1, 1,  2, 1, 1,  16, 16, 16,  16, 16, 16},
    { 64, 128, 16,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16},

    // MFMA 16x16x16, bk0=32 — for bk0max >= 64
    { 16,  32, 32,  1, 1, 1,  1, 1, 1,  16, 16, 16,  16, 16, 16},
    { 16,  64, 32,  1, 1, 1,  1, 1, 1,  16, 16, 16,  16, 16, 16},
    { 16, 128, 32,  1, 1, 1,  1, 1, 1,  16, 16, 16,  16, 16, 16},
    { 32,  64, 32,  2, 1, 1,  2, 1, 1,  16, 16, 16,  16, 16, 16},
    { 32, 128, 32,  2, 1, 1,  2, 1, 1,  16, 16, 16,  16, 16, 16},
    { 64,  32, 32,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16},
    { 64,  64, 32,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16},
    { 64, 128, 32,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16},
    {128,  64, 32,  8, 1, 1,  8, 1, 1,  16, 16, 16,  16, 16, 16},
    {128, 128, 32,  8, 1, 1,  8, 1, 1,  16, 16, 16,  16, 16, 16},
    // MFMA 32x32x16, bk0=32
    { 32,  64, 32,  1, 1, 1,  1, 1, 1,  32, 32, 16,  32, 32, 16},
    { 32, 128, 32,  1, 1, 1,  1, 1, 1,  32, 32, 16,  32, 32, 16},
    { 64,  64, 32,  2, 1, 1,  2, 1, 1,  32, 32, 16,  32, 32, 16},
    { 64, 128, 32,  2, 1, 1,  2, 1, 1,  32, 32, 16,  32, 32, 16},
    {128,  64, 32,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16},
    {128, 128, 32,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16},
    {256, 128, 32,  8, 1, 1,  8, 1, 1,  32, 32, 16,  32, 32, 16},
    // MFMA 16x16x32 for GEMM0 / 16x16x16 for GEMM1, bk0=32
    { 32, 128, 32,  2, 1, 1,  2, 1, 1,  16, 16, 32,  16, 16, 16},
    { 64, 128, 32,  4, 1, 1,  4, 1, 1,  16, 16, 32,  16, 16, 16},
    {128, 128, 32,  8, 1, 1,  8, 1, 1,  16, 16, 32,  16, 16, 16},
    // MFMA 16x16x32 wk0=32, bk0=32, rm0=1 (NWARP_SSHUFFLE eligible)
    { 16, 128, 32,  1, 1, 1,  1, 1, 1,  16, 16, 32,  16, 16, 16},
    // MFMA 16x16x32 wk0=32 (both gemms), bk0=32, rm0=1
    { 16,  64, 32,  1, 1, 1,  1, 1, 1,  16, 16, 32,  16, 16, 32},

    // MFMA 16x16x16, bk0=64 — for bk0max >= 128
    { 16,  64, 64,  1, 1, 1,  1, 1, 1,  16, 16, 32,  16, 16, 16},
    { 16, 128, 64,  1, 1, 1,  1, 1, 1,  16, 16, 32,  16, 16, 16},
    { 32, 128, 64,  2, 1, 1,  2, 1, 1,  16, 16, 32,  16, 16, 16},
    { 64, 128, 64,  4, 1, 1,  4, 1, 1,  16, 16, 32,  16, 16, 16},
    {128, 128, 64,  8, 1, 1,  8, 1, 1,  16, 16, 32,  16, 16, 16},
};
// clang-format on

static std::vector<device_fmha_common::TileConfig>
GenerateTileConfigs(std::size_t bk0max, std::size_t bn1,
                    const std::string& arch, DataType v_dtype)
{
    std::vector<device_fmha_common::TileConfig> tiles;
    constexpr std::size_t bk1 = 32;

    for(const auto& s : gfx9_base_shapes)
    {
        // bk0max must be divisible by bk0
        if(bk0max % s.bk0 != 0)
            continue;

        // k0_loops >= 2 for software pipelining
        if(bk0max < 2 * s.bk0)
            continue;

        // Limit inner loop ratio to avoid too many small iterations
        if(bk0max / s.bk0 > 8)
            continue;

        if(!IsValidVLdsConfig(s.rm0, bn1, bk1, arch, v_dtype))
            continue;

        tiles.push_back({s.bm0, s.bn0, s.bk0, bn1, bk1, bk0max,
                         s.rm0, s.rn0, s.rk0,
                         s.rm1, s.rn1, s.rk1,
                         s.wm0, s.wn0, s.wk0,
                         s.wm1, s.wn1, s.wk1});
    }

    return tiles;
}

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    std::vector<Operation> result;

    if(prob.dtype != DataType::Half)
        return result;

    if(!IsGfx9(arch) && !IsGfx950(arch))
        return result;

    std::size_t bk0max = CeilToValidHdim(prob.K);
    std::size_t bn1    = CeilToValidHdim(prob.O);

    if(bk0max == 0 || bn1 == 0)
        return result;

    auto tiles = GenerateTileConfigs(bk0max, bn1, arch, prob.dtype);
    if(tiles.empty())
        return result;

    bool needs_pad_k = (prob.K != bk0max);
    bool needs_pad_o = (prob.O != bn1);

    bool merge_heads =
        (bk0max == 128) && (prob.M == 1) && (prob.nhead_k < prob.nhead);

    for(const auto& tile : tiles)
    {
        bool needs_pad_m = (prob.M % tile.bm0 != 0);
        bool needs_pad_n = (prob.N % tile.bn0 != 0);

        bool has_uneven_splits = (prob.N % (tile.bn0 * prob.num_splits) != 0);

        for(const auto& pipeline_name : {"qr", "qr_nwarp_sshuffle"})
        {
            if(std::string(pipeline_name) == "qr_nwarp_sshuffle" && tile.rm0 != 1)
                continue;

            if(std::string(pipeline_name) == "qr" && tile.wk0 > 16 && tile.bk0 < 2 * tile.wk0)
                continue;

            Operation op;
            op.tile                           = tile;
            op.pipeline                       = pipeline_name;
            op.is_v_rowmajor                  = prob.is_v_rowmajor;
            op.dtype                          = prob.dtype;
            op.o_acc_dtype                    = prob.o_acc_dtype;
            op.pad_m                          = needs_pad_m;
            op.pad_n                          = needs_pad_n;
            op.pad_k                          = needs_pad_k;
            op.pad_o                          = needs_pad_o;
            op.has_uneven_splits              = has_uneven_splits;
            op.merge_num_head_groups_seqlen_q = merge_heads;
            result.push_back(op);
        }
    }

    return result;
}

using device_fmha_common::ToDataTypeString;

Solution Operation::ToSolution() const
{
    std::unordered_map<std::string, std::string> values = {
        {"DataType", ToDataTypeString(dtype)},
        {"OaccOutType", ToDataTypeString(o_acc_dtype)},

        {"BM0", std::to_string(tile.bm0)},
        {"BN0", std::to_string(tile.bn0)},
        {"BK0", std::to_string(tile.bk0)},
        {"BN1", std::to_string(tile.bn1)},
        {"BK1", std::to_string(tile.bk1)},
        {"BK0Max", std::to_string(tile.bk0max)},

        {"RM0", std::to_string(tile.rm0)},
        {"RN0", std::to_string(tile.rn0)},
        {"RK0", std::to_string(tile.rk0)},

        {"RM1", std::to_string(tile.rm1)},
        {"RN1", std::to_string(tile.rn1)},
        {"RK1", std::to_string(tile.rk1)},

        {"WM0", std::to_string(tile.wm0)},
        {"WN0", std::to_string(tile.wn0)},
        {"WK0", std::to_string(tile.wk0)},

        {"WM1", std::to_string(tile.wm1)},
        {"WN1", std::to_string(tile.wn1)},
        {"WK1", std::to_string(tile.wk1)},

        {"IsVRowMajor", is_v_rowmajor ? "true" : "false"},

        {"PadM", pad_m ? "true" : "false"},
        {"PadN", pad_n ? "true" : "false"},
        {"PadK", pad_k ? "true" : "false"},
        {"PadO", pad_o ? "true" : "false"},

        {"HasUnevenSplits", has_uneven_splits ? "true" : "false"},
        {"MergeNumHeadGroupsSeqLenQ", merge_num_head_groups_seqlen_q ? "true" : "false"},

        {"PipelineTag", pipeline == "qr_nwarp_sshuffle" ? "QR_NWARP_SSHUFFLE" : "QR"},
    };

    return Solution{InterpolateString(FmhaFwdSplitKVWrapperTemplate, values), std::move(values)};
}

} // namespace device_fmha_splitkv
} // namespace host
} // namespace ck
