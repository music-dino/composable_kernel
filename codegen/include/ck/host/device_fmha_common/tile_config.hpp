// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_common {

inline std::string ToDataTypeString(DataType dtype)
{
    switch(dtype)
    {
    case DataType::Half: return "ck_tile::fp16_t";
    case DataType::Float: return "float";
    default: return "ck_tile::fp16_t";
    }
}

// Tile configuration shared by FMHA kernels (fmha_fwd, fmha_splitkv, etc.)
struct TileConfig
{
    // Block tile
    std::size_t bm0;
    std::size_t bn0;
    std::size_t bk0;
    std::size_t bn1;
    std::size_t bk1;
    std::size_t bk0max;

    // Gemm0 block warps
    std::size_t rm0;
    std::size_t rn0;
    std::size_t rk0;

    // Gemm1 block warps
    std::size_t rm1;
    std::size_t rn1;
    std::size_t rk1;

    // Gemm0 warp tile
    std::size_t wm0;
    std::size_t wn0;
    std::size_t wk0;

    // Gemm1 warp tile
    std::size_t wm1;
    std::size_t wn1;
    std::size_t wk1;
};

// Result of looking up tile configurations for a given head dimension
struct HdimBucketResult
{
    std::size_t bucket_hdim   = 0;
    std::size_t bucket_hdim_v = 0;
    std::vector<TileConfig> tiles;
};

// Map from (hdim_q, hdim_v) bucket to tile configurations
using TileMap = std::map<std::pair<std::size_t, std::size_t>, std::vector<TileConfig>>;

} // namespace device_fmha_common
} // namespace host
} // namespace ck
