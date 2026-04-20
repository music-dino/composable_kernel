// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_splitkv/operation.hpp"
#include "ck/host/device_fmha_splitkv/problem.hpp"
#include "ck/host/stringutils.hpp"
#include <string>
#include <vector>
#include <iostream>

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
    "${HasUnevenSplits}, ${MergeNumHeadGroupsSeqLenQ}, "
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
    {{32, 32},   {{128,  64,  16,  32,  32,  32,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64,  64,  16,  32,  32,  32,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64,  64,  16,  32,  32,  32,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32,  64,  16,  32,  32,  32,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 16,  32,  16,  32,  32,  32,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128,  64,  16,  32,  32,  32,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  // bn0=32 variants
                  { 32,  32,  16,  32,  32,  32,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64,  32,  16,  32,  32,  32,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  // rm0=1 with bn0=64
                  { 16,  64,  16,  32,  32,  32,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16}}},
    //
    {{64, 64},   {{128,  64,  32,  64,  32,  64,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64,  64,  32,  64,  32,  64,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32,  64,  32,  64,  32,  64,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64,  64,  32,  64,  32,  64,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128,  64,  32,  64,  32,  64,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 16,  64,  32,  64,  32,  64,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32,  64,  32,  64,  32,  64,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16,  64,  32,  64,  32,  64,   1,  1,  1,   1,  1,  1,  16, 16, 32,  16, 16, 32},
                  // bn0=32 variants
                  { 32,  32,  32,  64,  32,  64,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64,  32,  32,  64,  32,  64,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64,  32,  32,  64,  32,  64,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16,  32,  32,  64,  32,  64,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16}}},
    //
    {{80, 96},   {{128, 128,  16,  96,  32,  80,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16, 128,  16,  96,  32,  80,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  16,  96,  32,  80,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  16,  96,  32,  80,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64, 128,  16,  96,  32,  80,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  // bn0=64 variants
                  { 64,  64,  16,  96,  32,  80,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64,  64,  16,  96,  32,  80,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32,  64,  16,  96,  32,  80,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 16,  64,  16,  96,  32,  80,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  // rm0=1 with (32,32,16) warps
                  { 32, 128,  16,  96,  32,  80,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32,  64,  16,  96,  32,  80,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  // bn0=64, rm0=4 with (32,32,16)
                  {128,  64,  16,  96,  32,  80,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16}}},
    //
    {{96, 128},  {{128, 128,  32, 128,  32,  96,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16, 128,  32, 128,  32,  96,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 128,  32,  96,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 128,  32,  96,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64, 128,  32, 128,  32,  96,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128, 128,  32, 128,  32,  96,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  // from {80,96}: rm0=1 with (32,32,16) warps
                  { 32, 128,  32, 128,  32,  96,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32,  64,  32, 128,  32,  96,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  // from {80,96}: bn0=64 variants
                  { 64,  64,  32, 128,  32,  96,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64,  64,  32, 128,  32,  96,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32,  64,  32, 128,  32,  96,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 16,  64,  32, 128,  32,  96,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128,  64,  32, 128,  32,  96,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16}}},
    //
    {{128, 128}, {{ 64, 128,  32, 128,  32, 128,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128,  64,  32, 128,  16, 128,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  {128, 128,  32, 128,  32, 128,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  // MFMA 16x16x16 variants
                  { 32, 128,  32, 128,  32, 128,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 128,  32, 128,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128, 128,  32, 128,  32, 128,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  // MFMA 32x32x16 variants
                  { 64, 128,  32, 128,  32, 128,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  // MFMA 16x16x32 for GEMM0, 16x16x16 for GEMM1 (wk1=32 produces invalid results)
                  { 32, 128,  64, 128,  32, 128,   2,  1,  1,   2,  1,  1,  16, 16, 32,  16, 16, 16},
                  { 64, 128,  64, 128,  32, 128,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128, 128,  64, 128,  32, 128,   8,  1,  1,   8,  1,  1,  16, 16, 32,  16, 16, 16},
                  // rm0=1 variants (eligible for QR_NWARP_SSHUFFLE)
                  { 16, 128,  32, 128,  32, 128,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 128,  32, 128,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16, 128,  64, 128,  32, 128,   1,  1,  1,   1,  1,  1,  16, 16, 32,  16, 16, 16},
                  { 16,  64,  32, 128,  32, 128,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32,  64,  32, 128,  32, 128,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16,  64,  64, 128,  32, 128,   1,  1,  1,   1,  1,  1,  16, 16, 32,  16, 16, 16}}},
    //
    {{192, 128}, {{128, 128,  32, 128,  32, 192,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64, 128,  32, 128,  32, 192,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  // MFMA 16x16x16 variants (rm0=1,2,4,8)
                  { 16, 128,  32, 128,  32, 192,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 128,  32, 192,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 128,  32, 192,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128, 128,  32, 128,  32, 192,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  // MFMA 32x32x16 variants (rm0=2,8)
                  { 64, 128,  32, 128,  32, 192,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  {256, 128,  32, 128,  32, 192,   8,  1,  1,   8,  1,  1,  32, 32, 16,  32, 32, 16},
                  // MFMA 16x16x32 for GEMM0, 16x16x16 for GEMM1 (bk0=32, rm0=2,8)
                  { 32, 128,  32, 128,  32, 192,   2,  1,  1,   2,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128, 128,  32, 128,  32, 192,   8,  1,  1,   8,  1,  1,  16, 16, 32,  16, 16, 16},
                  // MFMA 16x16x32 for GEMM0, 16x16x16 for GEMM1 (bk0=64, rm0=2,4,8)
                  { 32, 128,  64, 128,  32, 192,   2,  1,  1,   2,  1,  1,  16, 16, 32,  16, 16, 16},
                  { 64, 128,  64, 128,  32, 192,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128, 128,  64, 128,  32, 192,   8,  1,  1,   8,  1,  1,  16, 16, 32,  16, 16, 16},
                  // from {128,128}: rm0=1 variants
                  { 32, 128,  32, 128,  32, 192,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16, 128,  64, 128,  32, 192,   1,  1,  1,   1,  1,  1,  16, 16, 32,  16, 16, 16},
                  // from {128,128}: bn0=64 variants
                  { 16,  64,  32, 128,  32, 192,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32,  64,  32, 128,  32, 192,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16,  64,  64, 128,  32, 192,   1,  1,  1,   1,  1,  1,  16, 16, 32,  16, 16, 16}}},
    //
    {{192, 192}, {// MFMA 32x32x16 (original + rm0=2,8)
                  {128, 128,  32, 192,  32, 192,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64, 128,  32, 192,  32, 192,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  {256, 128,  32, 192,  32, 192,   8,  1,  1,   8,  1,  1,  32, 32, 16,  32, 32, 16},
                  // MFMA 16x16x16 (rm0=1,2,4,8)
                  { 16, 128,  32, 192,  32, 192,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 192,  32, 192,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 192,  32, 192,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128, 128,  32, 192,  32, 192,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  // MFMA 16x16x32 for GEMM0, 16x16x16 for GEMM1 (bk0=32, rm0=2,4,8)
                  { 32, 128,  32, 192,  32, 192,   2,  1,  1,   2,  1,  1,  16, 16, 32,  16, 16, 16},
                  { 64, 128,  32, 192,  32, 192,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128, 128,  32, 192,  32, 192,   8,  1,  1,   8,  1,  1,  16, 16, 32,  16, 16, 16},
                  // MFMA 16x16x32 for GEMM0, 16x16x16 for GEMM1 (bk0=64, rm0=2,4,8)
                  { 32, 128,  64, 192,  32, 192,   2,  1,  1,   2,  1,  1,  16, 16, 32,  16, 16, 16},
                  { 64, 128,  64, 192,  32, 192,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128, 128,  64, 192,  32, 192,   8,  1,  1,   8,  1,  1,  16, 16, 32,  16, 16, 16},
                  // from {192,128}: rm0=1 variants
                  { 32, 128,  32, 192,  32, 192,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16, 128,  64, 192,  32, 192,   1,  1,  1,   1,  1,  1,  16, 16, 32,  16, 16, 16},
                  // from {192,128}: bn0=64 variants
                  { 16,  64,  32, 192,  32, 192,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32,  64,  32, 192,  32, 192,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16,  64,  64, 192,  32, 192,   1,  1,  1,   1,  1,  1,  16, 16, 32,  16, 16, 16}}},
    //
    {{256, 256}, {// MFMA 32x32x16 (original + rm0=2,8)
                  {128, 128,  32, 256,  32, 256,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64, 128,  32, 256,  32, 256,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  {256, 128,  32, 256,  32, 256,   8,  1,  1,   8,  1,  1,  32, 32, 16,  32, 32, 16},
                  // MFMA 16x16x16 (rm0=1,2,4,8)
                  { 16, 128,  32, 256,  32, 256,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 256,  32, 256,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 256,  32, 256,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128, 128,  32, 256,  32, 256,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  // MFMA 16x16x32 for GEMM0, 16x16x16 for GEMM1 (bk0=32, rm0=2,8)
                  { 32, 128,  32, 256,  32, 256,   2,  1,  1,   2,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128, 128,  32, 256,  32, 256,   8,  1,  1,   8,  1,  1,  16, 16, 32,  16, 16, 16},
                  // MFMA 16x16x32 for GEMM0, 16x16x16 for GEMM1 (bk0=64, rm0=4)
                  { 64, 128,  64, 256,  32, 256,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  // MFMA 32x32x16, rm0=2
                  { 64, 128,  32, 256,  32, 256,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  // rm0=1 variants (eligible for QR_NWARP_SSHUFFLE)
                  { 16, 128,  32, 256,  32, 256,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 256,  32, 256,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16, 128,  64, 256,  32, 256,   1,  1,  1,   1,  1,  1,  16, 16, 32,  16, 16, 16},
                  // bn0=64 variants
                  {128,  64,  32, 256,  32, 256,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32,  64,  32, 256,  32, 256,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64,  64,  32, 256,  32, 256,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128,  64,  32, 256,  32, 256,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64,  64,  32, 256,  32, 256,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16,  64,  32, 256,  32, 256,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32,  64,  32, 256,  32, 256,   1,  1,  1,   1,  1,  1,  32, 32, 16,  32, 32, 16}}},
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
            std::cout << key.first << " " << key.second << std::endl;
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

        // GQA decode optimization: merge head groups with seqlen_q
        // Applies when hdim=128, seqlen_q=1, nhead_k < nhead_q
        bool merge_heads = (bucket.bucket_hdim == 128) && (prob.M == 1) && (prob.nhead_k < prob.nhead);

        // Generate operations for pipeline variants, filtering invalid combinations
        for(const auto& pipeline_name : {"qr", "qr_nwarp_sshuffle"})
        {
            // QR_NWARP_SSHUFFLE pipeline requires MWarp == 1 (rm0 == 1)
            if(std::string(pipeline_name) == "qr_nwarp_sshuffle" && tile.rm0 != 1)
                continue;

            // Both pipelines require k0_loops >= 2 (bk0max >= 2 * bk0) for correct pipelining
            if(tile.bk0max < 2 * tile.bk0)
                continue;

            // QR pipeline produces incorrect results when wk0 > 16 and bk0 < 2 * wk0
            if(std::string(pipeline_name) == "qr" && tile.wk0 > 16 && tile.bk0 < 2 * tile.wk0)
                continue;

            Operation op;
            op.tile                           = tile;
            op.pipeline                       = pipeline_name;
            op.is_v_rowmajor                  = prob.is_v_rowmajor;
            op.dtype                          = prob.dtype;
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
