// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_appendkv/operation.hpp"
#include "ck/host/device_fmha_appendkv/problem.hpp"
#include "ck/host/device_fmha_common/tile_config.hpp"
#include "ck/host/stringutils.hpp"
#include <map>
#include <string>
#include <vector>

namespace ck {
namespace host {
namespace device_fmha_appendkv {

static const char* const FmhaFwdAppendKVWrapperTemplate =
    "ck_tile::FmhaFwdAppendKVWrapper<${DataType}, "
    "${M0}, ${N0}, ${K0}, ${N1}, "
    "${IsVRowMajor}, "
    "${PadSeqLenQ}, ${PadSeqLenK}, ${PadHdimQ}, ${PadHdimV}, "
    "${RotaryEnum}>";

static bool IsGfx9(const std::string& arch)
{
    return arch.find("gfx9") == 0 && arch.find("gfx950") != 0;
}

static bool IsGfx950(const std::string& arch) { return arch.find("gfx950") == 0; }

bool IsSupportedArch(const std::string& arch) { return IsGfx9(arch) || IsGfx950(arch); }

// Map from (hdim_q, hdim_v) bucket to tile configuration
using TileMap = std::map<std::pair<std::size_t, std::size_t>, TileConfig>;

// Tile configurations for AppendKV
// From fmha_fwd_appendkv.py: get_hdim_tile_size_dict
// Tile: {m0, n0, k0, n1} where m0=seqlen_q tile, n0=seqlen_knew tile, k0=hdim_q tile, n1=hdim_v
// tile
// clang-format off
static const TileMap gfx9_fp16_tiles = {
    //    hdim_q, hdim_v   m0  n0   k0   n1
    {{  32,  32}, { 64, 64,  32,  32}},
    {{  64,  64}, { 64, 64,  64,  64}},
    {{ 128, 128}, { 64, 64, 128, 128}},
    {{ 256, 256}, { 64, 64, 256, 256}},
};
// clang-format on

struct HdimBucketResult
{
    std::size_t bucket_hdim   = 0;
    std::size_t bucket_hdim_v = 0;
    TileConfig tile           = {};
};

static HdimBucketResult GetTileConfigForHdim(const std::string& arch,
                                             DataType dtype,
                                             std::size_t hdim_q,
                                             std::size_t hdim_v)
{
    HdimBucketResult result;

    if(dtype != DataType::Half)
        return result;

    if(!IsGfx9(arch) && !IsGfx950(arch))
        return result;

    for(const auto& [key, tile] : gfx9_fp16_tiles)
    {
        if(hdim_q <= key.first && hdim_v <= key.second)
        {
            result.bucket_hdim   = key.first;
            result.bucket_hdim_v = key.second;
            result.tile          = tile;
            return result;
        }
    }

    return result;
}

static std::string RotaryEnumToString(RotaryEmbeddingType rotary)
{
    switch(rotary)
    {
    case RotaryEmbeddingType::Interleaved: return "ck_tile::RotaryEmbeddingEnum::INTERLEAVED";
    case RotaryEmbeddingType::HalfRotated: return "ck_tile::RotaryEmbeddingEnum::HALF_ROTATED";
    default: return "ck_tile::RotaryEmbeddingEnum::NONE";
    }
}

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    std::vector<Operation> result;

    auto bucket = GetTileConfigForHdim(arch, prob.dtype, prob.K, prob.O);
    if(bucket.bucket_hdim == 0)
        return result;

    const auto& tile = bucket.tile;

    // Compute padding flags
    bool needs_pad_seqlen_q = (prob.M % tile.m0 != 0);
    bool needs_pad_seqlen_k = (prob.N % tile.n0 != 0); // seqlen_knew
    bool needs_pad_hdim_q   = (prob.K != bucket.bucket_hdim);
    bool needs_pad_hdim_v   = (prob.O != bucket.bucket_hdim_v);

    Operation op;
    op.tile          = tile;
    op.dtype         = prob.dtype;
    op.is_v_rowmajor = prob.is_v_rowmajor;
    op.rotary        = prob.rotary;
    op.pad_seqlen_q  = needs_pad_seqlen_q;
    op.pad_seqlen_k  = needs_pad_seqlen_k;
    op.pad_hdim_q    = needs_pad_hdim_q;
    op.pad_hdim_v    = needs_pad_hdim_v;

    result.push_back(op);

    return result;
}

using device_fmha_common::ToDataTypeString;

Solution Operation::ToSolution() const
{
    std::unordered_map<std::string, std::string> values = {
        {"DataType", ToDataTypeString(dtype)},

        {"M0", std::to_string(tile.m0)},
        {"N0", std::to_string(tile.n0)},
        {"K0", std::to_string(tile.k0)},
        {"N1", std::to_string(tile.n1)},

        {"IsVRowMajor", is_v_rowmajor ? "true" : "false"},

        {"PadSeqLenQ", pad_seqlen_q ? "true" : "false"},
        {"PadSeqLenK", pad_seqlen_k ? "true" : "false"},
        {"PadHdimQ", pad_hdim_q ? "true" : "false"},
        {"PadHdimV", pad_hdim_v ? "true" : "false"},

        {"RotaryEnum", RotaryEnumToString(rotary)},
    };

    return Solution{InterpolateString(FmhaFwdAppendKVWrapperTemplate, values), std::move(values)};
}

} // namespace device_fmha_appendkv
} // namespace host
} // namespace ck
