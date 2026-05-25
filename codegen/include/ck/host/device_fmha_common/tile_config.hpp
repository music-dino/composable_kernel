// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_common {

// --- Architecture helpers ---

inline bool IsGfx9(const std::string& arch)
{
    return arch.find("gfx9") == 0 && arch.find("gfx950") != 0;
}

inline bool IsGfx950(const std::string& arch) { return arch.find("gfx950") == 0; }

inline bool IsGfx12(const std::string& arch) { return arch.find("gfx12") == 0; }

inline bool IsSupportedFmhaArch(const std::string& arch)
{
    return IsGfx9(arch) || IsGfx950(arch);
}

inline std::size_t GetWarpSize(const std::string& /*arch*/) { return 64; }

inline std::size_t GetLdsBanks(const std::string& arch)
{
    if(IsGfx950(arch))
        return 64;
    return 32;
}

// --- Data type helpers ---

inline std::string ToDataTypeString(DataType dtype)
{
    switch(dtype)
    {
    case DataType::Half: return "ck_tile::fp16_t";
    case DataType::Float: return "float";
    default: return "ck_tile::fp16_t";
    }
}

inline std::size_t DataTypeSize(DataType dtype)
{
    switch(dtype)
    {
    case DataType::Half: return 2;
    case DataType::Float: return 4;
    case DataType::Int8: return 1;
    case DataType::Int32: return 4;
    default: return 0;
    }
}

// --- Head dimension helpers ---

// Valid head dimensions — must pass ceil_to_qualified_tile_length
// in tile_fmha_shape.hpp (powers of 2 + special cases 48, 80, 96, 160, 192)
inline constexpr std::size_t valid_hdims[] = {32, 48, 64, 80, 96, 128, 160, 192, 256};

inline std::size_t CeilToValidHdim(std::size_t dim)
{
    for(std::size_t h : valid_hdims)
    {
        if(dim <= h)
            return h;
    }
    return 0;
}

// --- Tile validity helpers ---

// Check whether a tile configuration produces a valid V LDS descriptor.
// When total_pixels = bn1 * bk1 / kBlockSize is not a power of 2,
// kKPack won't divide PixelsPerRow, causing static_assert failures
// in MakeVLdsBlockDescriptor (block_fmha_pipeline_qx_ks_vs_custom_policy.hpp).
inline bool IsValidVLdsConfig(std::size_t rm0,
                              std::size_t bn1,
                              std::size_t bk1,
                              const std::string& arch,
                              DataType v_dtype)
{
    std::size_t warp_size    = GetWarpSize(arch);
    std::size_t lds_banks    = GetLdsBanks(arch);
    std::size_t v_size       = DataTypeSize(v_dtype);
    std::size_t pixelsPerRow = lds_banks * 4 / v_size;
    std::size_t kBlockSize   = rm0 * warp_size;
    std::size_t total_pixels = bn1 * bk1 / kBlockSize;
    std::size_t kKPack       = std::min(total_pixels, 16 / v_size);

    // Mirrors static asserts in MakeVLdsBlockDescriptor.
    // The first check is the critical one; the other two are implied by it
    // for our current parameter space but included as defensive checks.
    if(pixelsPerRow % kKPack != 0)
        return false;
    std::size_t nPerRow = pixelsPerRow / kKPack;
    if(bn1 % nPerRow != 0)
        return false;
    if(bk1 % kKPack != 0)
        return false;

    return true;
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
