// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/device_fmha_common/tile_config.hpp"
#include "ck/host/device_fmha_splitkv/problem.hpp"

namespace ck {
namespace host {
namespace device_fmha_splitkv {

using TileConfig      = device_fmha_common::TileConfig;
using HdimBucketResult = device_fmha_common::HdimBucketResult;

struct Operation
{
    TileConfig tile = {};

    std::string pipeline = "qr"; // "qr" or "qr_nwarp_sshuffle"

    bool is_v_rowmajor = true;
    DataType dtype     = DataType::Half;

    bool pad_m = true; // pad seqlen_q
    bool pad_n = true; // pad seqlen_k
    bool pad_k = true; // pad hdim_q
    bool pad_o = true; // pad hdim_v

    bool has_uneven_splits = true; // whether splits may be uneven

    static std::vector<Operation> CreateOperations(const Problem& prob, const std::string& arch);

    Solution ToSolution() const;
};

HdimBucketResult
GetTileConfigsForHdim(const std::string& arch, DataType dtype, std::size_t K, std::size_t O);

bool IsSupportedArch(const std::string& arch);

} // namespace device_fmha_splitkv
} // namespace host
} // namespace ck
