// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// This header is designed to be embedded and used at RTC compilation time.

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"

namespace ck_tile {

template <typename DataType_,
          index_t kHeadDimV,
          index_t kN1,           // tile size for hdim_v
          index_t kLogMaxSplits, // log2 of max splits (3=8, 4=16, 5=32, 6=64, 7=128)
          bool kPadSeqLenQ,
          bool kPadHeadDimV>
struct FmhaFwdSplitKVCombineWrapper
{
    using FmhaTraits = TileFmhaFwdSplitKVCombineTraits<kPadSeqLenQ,
                                                       kPadHeadDimV,
                                                       true,   // kStoreLSE (always output LSE)
                                                       false,  // kDoFp8StaticQuant
                                                       kLogMaxSplits,
                                                       -1>;    // kBlockPerCu

    using PipelineProblem =
        BlockFmhaSplitKVCombinePipelineProblem<float,       // LSE type
                                               float,       // Oacc type
                                               DataType_,   // O type
                                               kHeadDimV,
                                               false,       // kIsGroupMode (batch mode only)
                                               kN1,
                                               FmhaTraits>;

    using Pipeline = BlockFmhaFwdSplitKVCombinePipeline<PipelineProblem>;

    using Epilogue = Default2DEpilogue<Default2DEpilogueProblem<float, DataType_, false, false>>;

    using Kernel = FmhaFwdSplitKVCombineKernel<Pipeline, Epilogue>;

    // Tensor layouts:
    // lse_acc: [batch, nhead, num_splits, M] (input from splitkv)
    // o_acc:   [batch, nhead, num_splits, M, O] (input from splitkv)
    // lse:     [batch, nhead, M] (output)
    // o:       [batch, nhead, M, O] (output)
    struct Descriptor
    {
        index_t batch, nhead, M, O;
        index_t num_splits;

        index_t lse_acc_stride_batch, lse_acc_stride_nhead, lse_acc_stride_split;
        index_t o_acc_stride_batch, o_acc_stride_nhead, o_acc_stride_split, o_acc_stride_m;

        index_t lse_stride_batch, lse_stride_nhead;
        index_t o_stride_batch, o_stride_nhead, o_stride_m;

        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return Kernel::kIsAvailable; }
    };

    template <typename LseAccStrides,
              typename OAccDims,
              typename OAccStrides,
              typename LseStrides,
              typename OStrides>
    CK_TILE_HOST_DEVICE static constexpr auto make_descriptor(index_t batch,
                                                              index_t nhead,
                                                              index_t seqlen_q,
                                                              index_t hdim_v,
                                                              index_t num_splits,
                                                              LseAccStrides lse_acc_strides,
                                                              OAccStrides o_acc_strides,
                                                              LseStrides lse_strides,
                                                              OStrides o_strides)
    {
        return Descriptor{batch,
                          nhead,
                          seqlen_q,
                          hdim_v,
                          num_splits,
                          //
                          lse_acc_strides[number<0>{}],
                          lse_acc_strides[number<1>{}],
                          lse_acc_strides[number<2>{}],
                          //
                          o_acc_strides[number<0>{}],
                          o_acc_strides[number<1>{}],
                          o_acc_strides[number<2>{}],
                          o_acc_strides[number<3>{}],
                          //
                          lse_strides[number<0>{}],
                          lse_strides[number<1>{}],
                          //
                          o_strides[number<0>{}],
                          o_strides[number<1>{}],
                          o_strides[number<2>{}]};
    }

    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   const float* lse_acc_ptr,
                                   const float* o_acc_ptr,
                                   float* lse_ptr,
                                   DataType_* o_ptr)
    {
        using Kargs = typename Kernel::Kargs;
        Kargs kargs{};

        kargs.lse_acc_ptr = lse_acc_ptr;
        kargs.o_acc_ptr   = o_acc_ptr;
        kargs.o_ptr       = o_ptr;

        kargs.batch      = desc.batch;
        kargs.seqlen_q   = desc.M;
        kargs.hdim_v     = desc.O;
        kargs.num_splits = desc.num_splits;

        kargs.row_stride_o_acc = desc.o_acc_stride_m;
        kargs.row_stride_o     = desc.o_stride_m;

        kargs.nhead_stride_lse_acc = desc.lse_acc_stride_nhead;
        kargs.nhead_stride_o_acc   = desc.o_acc_stride_nhead;
        kargs.nhead_stride_o       = desc.o_stride_nhead;

        kargs.split_stride_lse_acc = desc.lse_acc_stride_split;
        kargs.split_stride_o_acc   = desc.o_acc_stride_split;

        // LSE output (always enabled with kStoreLSE = true)
        kargs.lse_ptr          = lse_ptr;
        kargs.nhead_stride_lse = desc.lse_stride_nhead;
        kargs.batch_stride_lse = desc.lse_stride_batch;

        kargs.batch_stride_lse_acc = desc.lse_acc_stride_batch;
        kargs.batch_stride_o_acc   = desc.o_acc_stride_batch;
        kargs.batch_stride_o       = desc.o_stride_batch;

        Kernel{}(kargs);
    }
};

} // namespace ck_tile
