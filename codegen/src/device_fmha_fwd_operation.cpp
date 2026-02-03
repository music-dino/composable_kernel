// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_fwd/operation.hpp"
#include "ck/host/device_fmha_fwd/problem.hpp"
#include "ck/host/stringutils.hpp"
#include <map>
#include <string>
#include <vector>

namespace ck {
namespace host {
namespace device_fmha_fwd {

static const char* const FmhaFwdWrapperTemplate =
    "ck_tile::FmhaFwdWrapper<${DataType}, "
    "${BM0}, ${BN0}, ${BK0}, ${BN1}, ${BK1}, ${BK0Max}, "
    "${RM0}, ${RN0}, ${RK0}, ${RM1}, ${RN1}, ${RK1}, "
    "${WM0}, ${WN0}, ${WK0}, ${WM1}, ${WN1}, ${WK1}, "
    "${IsCausal}, ${IsVRowMajor}, ${HasBias}, "
    "${PadM}, ${PadN}, ${PadK}, ${PadO}, "
    "ck_tile::FmhaPipelineTag::${PipelineTag}>";

static bool IsGfx950(const std::string& arch) { return arch.find("gfx950") == 0; }
static bool IsGfx12(const std::string& arch) { return arch.find("gfx12") == 0; }

using TileMap = std::map<std::pair<std::size_t, std::size_t>, std::vector<TileConfig>>;

// gfx9 fp16/bf16 tiles from KernelComponentFactoryGfx9::get_hdim_tile_size_dict
// clang-format off
static const TileMap gfx9_fp16_tiles = {
    //             bm0, bn0, bk0, bn1, bk1,bk0max,rm0,rn0,rk0,rm1,rn1,rk1, wm0,wn0,wk0, wm1,wn1,wk1
    {{32, 32},   {{128,  64,  16,  32,  32,  32,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16}}},
    {{64, 64},   {{ 16,  32,  64,  64,  32,  64,   1,  1,  1,   1,  1,  1,  16, 16, 32,  16, 16, 32},
                  { 32,  32,  64,  64,  32,  64,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  {128,  64,  32,  64,  32,  64,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16}}},
    {{80, 96},   {{128, 128,  16,  96,  32,  80,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16}}},
    {{96, 128},  {{128, 128,  32, 128,  32,  96,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16}}},
    {{128, 128}, {{ 16,  32,  64, 128,  32, 128,   1,  1,  1,   1,  1,  1,  16, 16, 32,  16, 16, 32},
                  { 32,  32, 128, 128,  32, 128,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64, 128,  32, 128,  32, 128,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128,  64,  32, 128,  16, 128,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  {128, 128,  32, 128,  32, 128,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16}}},
    {{192, 128}, {{128, 128,  32, 128,  32, 192,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16}}},
    {{192, 192}, {{128, 128,  32, 192,  32, 192,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16}}},
    {{256, 256}, {{128, 128,  32, 256,  32, 256,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16}}},
};

// gfx12 fp16/bf16 tiles from KernelComponentFactoryGfx12::get_hdim_tile_size_dict
static const TileMap gfx12_fp16_tiles = {
    //             bm0, bn0, bk0, bn1, bk1,bk0max,rm0,rn0,rk0,rm1,rn1,rk1, wm0,wn0,wk0, wm1,wn1,wk1
    {{32, 32},   {{ 64,  64,  16,  32,  32,  32,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{64, 64},   {{ 64,  64,  32,  64,  32,  64,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{128, 128}, {{ 64,  64,  32, 128,  32, 128,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{192, 128}, {{ 64,  64,  32, 128,  32, 256,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{256, 256}, {{ 64,  64,  32, 256,  32, 256,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
};
// clang-format on

HdimBucketResult
GetTileConfigsForHdim(const std::string& arch, DataType dtype, std::size_t K, std::size_t O)
{
    HdimBucketResult result;

    if(dtype != DataType::Half)
        return result;

    const TileMap& tile_map = IsGfx12(arch) ? gfx12_fp16_tiles : gfx9_fp16_tiles;

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

struct PipelineConfig
{
    std::string name;
    bool pad_m;
    bool pad_n;
    bool pad_k;
    bool pad_o;
};

static std::vector<PipelineConfig> GetPipelinesGfx12()
{
    return {
        {"qr", false, false, false, false},
        {"qr", true, true, true, true},
    };
}

static std::vector<PipelineConfig>
GetPipelinesGfx9(std::size_t bucket_hdim, std::size_t bucket_hdim_v, bool has_bias)
{
    std::vector<PipelineConfig> configs;

    if(bucket_hdim == 256 && bucket_hdim_v == 256)
    {
        configs.push_back({"qr", false, false, false, false});
        configs.push_back({"qr", true, true, false, false});
        configs.push_back({"qr", true, true, true, true});
    }
    else if(has_bias)
    {
        configs.push_back({"qr", false, false, false, false});
        configs.push_back({"qr", true, true, true, true});
    }
    else
    {
        configs.push_back({"qr_async", true, false, true, true});
        configs.push_back({"qr_async", true, true, true, true});
        configs.push_back({"qr", true, true, true, true});
    }

    return configs;
}

static std::vector<PipelineConfig>
GetPipelinesGfx950(std::size_t bucket_hdim, std::size_t bucket_hdim_v, bool has_bias)
{
    auto configs = GetPipelinesGfx9(bucket_hdim, bucket_hdim_v, has_bias);

    bool is_hdim_256 = (bucket_hdim == 256 && bucket_hdim_v == 256);
    if(!is_hdim_256 && !has_bias)
    {
        configs.push_back({"qr_async_trload", false, false, false, false});
        configs.push_back({"qr_async_trload", false, false, true, true});
    }

    return configs;
}

static std::vector<PipelineConfig> GetPipelineConfigs(const std::string& arch,
                                                      std::size_t bucket_hdim,
                                                      std::size_t bucket_hdim_v,
                                                      bool has_bias)
{
    if(IsGfx12(arch))
        return GetPipelinesGfx12();
    if(IsGfx950(arch))
        return GetPipelinesGfx950(bucket_hdim, bucket_hdim_v, has_bias);
    return GetPipelinesGfx9(bucket_hdim, bucket_hdim_v, has_bias);
}

static bool IsPaddingCompatible(const PipelineConfig& config,
                                const Problem& prob,
                                const TileConfig& tile,
                                std::size_t bucket_hdim,
                                std::size_t bucket_hdim_v)
{
    bool needs_pad_m = (prob.M % tile.bm0 != 0);
    bool needs_pad_n = (prob.N % tile.bn0 != 0);
    bool needs_pad_k = (prob.K != bucket_hdim);
    bool needs_pad_o = (prob.O != bucket_hdim_v);

    // +------------+----------+------------+
    // | config.pad | needs_pad| compatible |
    // +------------+----------+------------+
    // |   false    |  false   |    true    |
    // |   false    |  true    |    false   |
    // |   true     |  false   |    true    |
    // |   true     |  true    |    true    |
    // +------------+----------+------------+
    //
    return (config.pad_m || !needs_pad_m) && (config.pad_n || !needs_pad_n) &&
           (config.pad_k || !needs_pad_k) && (config.pad_o || !needs_pad_o);
}

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    std::vector<Operation> result;

    auto bucket = GetTileConfigsForHdim(arch, prob.dtype, prob.K, prob.O);
    auto pipelines =
        GetPipelineConfigs(arch, bucket.bucket_hdim, bucket.bucket_hdim_v, prob.has_bias);

    for(const auto& tile : bucket.tiles)
    {
        for(const auto& pipeline : pipelines)
        {
            if(!IsGfx12(arch) && prob.dtype != DataType::Float)
            {
                bool is_bucket_128 = (bucket.bucket_hdim == 128 && bucket.bucket_hdim_v == 128);

                if(is_bucket_128)
                {
                    if(tile.bn0 != 128)
                        continue;
                    if(pipeline.name != "qr_async" && tile.bk0 == 64)
                        continue;
                }
                else
                {
                    if(tile.bm0 != 128)
                        continue;
                }
            }

            if(pipeline.name == "qr_async" || pipeline.name == "qr_async_trload")
            {
                if(prob.dtype == DataType::Half && (prob.K % 8 != 0 || prob.O % 8 != 0))
                    continue;
            }

            if(!IsPaddingCompatible(pipeline, prob, tile, bucket.bucket_hdim, bucket.bucket_hdim_v))
                continue;

            Operation op;
            op.tile = tile;

            op.pipeline      = pipeline.name;
            op.is_causal     = prob.is_causal;
            op.is_v_rowmajor = prob.is_v_rowmajor;
            op.has_bias      = prob.has_bias;
            op.dtype         = prob.dtype;
            op.pad_m         = pipeline.pad_m;
            op.pad_n         = pipeline.pad_n;
            op.pad_k         = pipeline.pad_k;
            op.pad_o         = pipeline.pad_o;

            result.push_back(op);
        }
    }

    return result;
}

static std::string ToDataTypeString(DataType dtype)
{
    switch(dtype)
    {
    case DataType::Half: return "ck_tile::fp16_t";
    case DataType::Float: return "float";
    default: return "ck_tile::fp16_t";
    }
}

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

        {"IsCausal", is_causal ? "true" : "false"},
        {"IsVRowMajor", is_v_rowmajor ? "true" : "false"},
        {"HasBias", has_bias ? "true" : "false"},

        {"PadM", pad_m ? "true" : "false"},
        {"PadN", pad_n ? "true" : "false"},
        {"PadK", pad_k ? "true" : "false"},
        {"PadO", pad_o ? "true" : "false"},

        {"PipelineTag",
         pipeline == "qr_async_trload" ? "QR_ASYNC_TRLOAD"
                                       : (pipeline == "qr_async" ? "QR_ASYNC" : "QR")},
    };

    return Solution{InterpolateString(FmhaFwdWrapperTemplate, values), std::move(values)};
}

} // namespace device_fmha_fwd
} // namespace host
} // namespace ck
