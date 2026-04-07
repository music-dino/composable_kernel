// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_splitkv/operation.hpp"
#include "ck/host/device_fmha_splitkv/problem.hpp"
#include "ck/host/stringutils.hpp"
#include <string>
#include <vector>

namespace ck {
namespace host {
namespace device_fmha_splitkv {

static const char* const FmhaFwdSplitKVWrapperTemplate =
    "ck_tile::FmhaFwdSplitKVWrapper<${DataType}, "
    "${BM0}, ${BN0}, ${BK0}, ${BN1}, ${BK1}, ${BK0Max}, "
    "${RM0}, ${RN0}, ${RK0}, ${RM1}, ${RN1}, ${RK1}, "
    "${WM0}, ${WN0}, ${WK0}, ${WM1}, ${WN1}, ${WK1}, "
    "${IsVRowMajor}, "
    "${PadM}, ${PadN}, ${PadK}, ${PadO}, "
    "${HasUnevenSplits}, "
    "ck_tile::FmhaSplitKVPipelineTag::${PipelineTag}>";

static bool IsGfx9(const std::string& arch)
{
    return arch.find("gfx9") == 0 && arch.find("gfx950") != 0;
}

static bool IsGfx950(const std::string& arch) { return arch.find("gfx950") == 0; }

bool IsSupportedArch(const std::string& arch)
{
    return IsGfx9(arch) || IsGfx950(arch);
}

using TileMap = device_fmha_common::TileMap;

// gfx9 fp16/bf16 tile configurations for SplitKV
// From fmha_fwd_splitkv.py KernelComponentFactoryGfx9::get_hdim_tile_size_dict
//
// clang-format off
static const TileMap gfx9_fp16_tiles = {
    //             bm0, bn0, bk0, bn1, bk1,bk0max,rm0,rn0,rk0,rm1,rn1,rk1, wm0,wn0,wk0, wm1,wn1,wk1
    {{32, 32},   {{ 32,  64,  16,  32,  32,  32,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{64, 64},   {{ 64,  64,  32,  64,  32,  64,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{96, 128},  {{ 64, 128,  32, 128,  32,  96,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{128, 128}, {{ 64, 128,  32, 128,  32, 128,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{256, 256}, {{ 64, 128,  32, 256,  32, 256,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
};
// clang-format on

HdimBucketResult
GetTileConfigsForHdim(const std::string& arch, DataType dtype, std::size_t K, std::size_t O)
{
    HdimBucketResult result;

    if(dtype != DataType::Half)
        return result;

    if(!IsGfx9(arch) && !IsGfx950(arch))
        return result;

    const TileMap& tile_map = gfx9_fp16_tiles;

    for(const auto& [key, tiles] : tile_map)
    {
        if(K <= key.first && O <= key.second)
        {
            result.bucket_hdim   = key.first;
            result.bucket_hdim_v = key.second;
            result.tiles         = tiles;
            return result;
        }
    }

    return result;
}

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    std::vector<Operation> result;

    auto bucket = GetTileConfigsForHdim(arch, prob.dtype, prob.K, prob.O);
    if(bucket.tiles.empty())
        return result;

    for(const auto& tile : bucket.tiles)
    {
        // Compute exact padding needs for this tile
        bool needs_pad_m = (prob.M % tile.bm0 != 0);
        bool needs_pad_n = (prob.N % tile.bn0 != 0);
        bool needs_pad_k = (prob.K != bucket.bucket_hdim);
        bool needs_pad_o = (prob.O != bucket.bucket_hdim_v);

        // Check if seqlen_k is evenly divisible across splits and tiles
        bool has_uneven_splits = (prob.N % (tile.bn0 * prob.num_splits) != 0);

        // Generate operations for pipeline variants, filtering invalid combinations
        for(const auto& pipeline_name : {"qr", "qr_nwarp_sshuffle"})
        {
            // QR_NWARP_SSHUFFLE pipeline requires MWarp == 1 (rm0 == 1)
            if(std::string(pipeline_name) == "qr_nwarp_sshuffle" && tile.rm0 != 1)
                continue;

            Operation op;
            op.tile              = tile;
            op.pipeline          = pipeline_name;
            op.is_v_rowmajor     = prob.is_v_rowmajor;
            op.dtype             = prob.dtype;
            op.pad_m             = needs_pad_m;
            op.pad_n             = needs_pad_n;
            op.pad_k             = needs_pad_k;
            op.pad_o             = needs_pad_o;
            op.has_uneven_splits = has_uneven_splits;
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

        {"PipelineTag", pipeline == "qr_nwarp_sshuffle" ? "QR_NWARP_SSHUFFLE" : "QR"},
    };

    return Solution{InterpolateString(FmhaFwdSplitKVWrapperTemplate, values), std::move(values)};
}

} // namespace device_fmha_splitkv
} // namespace host
} // namespace ck
