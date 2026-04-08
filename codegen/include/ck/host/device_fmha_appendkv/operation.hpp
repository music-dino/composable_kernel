// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/host/device_fmha_appendkv/problem.hpp"
#include "ck/host/device_fmha_common/tile_config.hpp"
#include "ck/host/types.hpp"
#include <string>
#include <vector>

namespace ck {
namespace host {
namespace device_fmha_appendkv {

struct TileConfig
{
    std::size_t m0 = 0; // seqlen_q tile
    std::size_t n0 = 0; // seqlen_knew tile
    std::size_t k0 = 0; // hdim_q tile
    std::size_t n1 = 0; // hdim_v tile
};

struct Operation
{
    TileConfig tile = {};

    DataType dtype             = DataType::Half;
    bool is_v_rowmajor         = true;
    RotaryEmbeddingType rotary = RotaryEmbeddingType::None;

    bool pad_seqlen_q = false;
    bool pad_seqlen_k = false;
    bool pad_hdim_q   = false;
    bool pad_hdim_v   = false;

    static std::vector<Operation> CreateOperations(const Problem& prob, const std::string& arch);

    Solution ToSolution() const;
};

} // namespace device_fmha_appendkv
} // namespace host
} // namespace ck
