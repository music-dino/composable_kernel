// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// This header is designed to be embedded and used at RTC compilation time.

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"

namespace ck_tile {

enum class FmhaSplitKVPipelineTag
{
    QR,               // BlockFmhaFwdSplitKVPipelineQRKSVS
    QR_NWARP_SSHUFFLE // BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS
};

template <typename DataType_,
          // Block tile
          index_t kBM0,
          index_t kBN0,
          index_t kBK0,
          index_t kBN1,
          index_t kBK1,
          index_t kBK0Max,
          // Gemm0 block warps
          index_t kRM0,
          index_t kRN0,
          index_t kRK0,
          // Gemm1 block warps
          index_t kRM1,
          index_t kRN1,
          index_t kRK1,
          // Gemm0 warp tile
          index_t kWM0,
          index_t kWN0,
          index_t kWK0,
          // Gemm1 warp tile
          index_t kWM1,
          index_t kWN1,
          index_t kWK1,
          //
          bool kIsVRowMajor,
          //
          bool kPadM,
          bool kPadN,
          bool kPadK,
          bool kPadO,
          //
          bool kHasUnevenSplits,
          //
          FmhaSplitKVPipelineTag kPipelineTag>
struct FmhaFwdSplitKVWrapper
{
    using BlockTile = sequence<kBM0, kBN0, kBK0, kBN1, kBK1, kBK0Max>;

    using Gemm0BlockWarps = sequence<kRM0, kRN0, kRK0>;
    using Gemm0WarpTile   = sequence<kWM0, kWN0, kWK0>;
    using Gemm1BlockWarps = sequence<kRM1, kRN1, kRK1>;
    using Gemm1WarpTile   = sequence<kWM1, kWN1, kWK1>;

    using FmhaShape = TileFmhaShape<BlockTile,
                                    Gemm0BlockWarps,
                                    Gemm0WarpTile,
                                    Gemm1BlockWarps,
                                    Gemm1WarpTile,
                                    kIsVRowMajor>;

    using FmhaTraits = TileFmhaFwdSplitKVTraits<kPadM, // kPadSeqLenQ
                                                kPadN, // kPadSeqLenK
                                                kPadK, // kPadHeadDimQ
                                                kPadO, // kPadHeadDimV
                                                false, // kHasLogitsSoftCap
                                                BlockAttentionBiasEnum::NO_BIAS,
                                                false, // kHasBiasGrad
                                                true,  // kStoreLSE (always true for splitkv)
                                                false, // kDoFp8StaticQuant
                                                false, // kIsPagedKV
                                                kHasUnevenSplits,
                                                // TODO: Add support for kMergeNumHeadGroupsSeqLenQ
                                                false,  // kMergeNumHeadGroupsSeqLenQ
                                                -1,     // kBlockPerCu
                                                false>; // kHasSink

    using FmhaMask = SimplifiedGenericAttentionMask<false>; // No masking

    using PipelineProblem =
        BlockFmhaFwdSplitKVPipelineProblem<DataType_, // Q type
                                           DataType_, // K type
                                           DataType_, // V type
                                           // TODO: Not necessarily always float
                                           float,     // Sacc type
                                           float,     // SMPLCompute type
                                           DataType_, // Bias type
                                           float,     // LSE type
                                           DataType_, // P type
                                           float,     // Oacc type
                                           float,     // OaccOut type (workspace output)
                                           FmhaShape,
                                           false, // kIsGroupMode (batch mode only)
                                           ComposedAttention<false, CK_TILE_FMHA_FWD_FAST_EXP2>,
                                           FmhaMask,
                                           FmhaTraits>;

    using Pipeline =
        std::conditional_t<kPipelineTag == FmhaSplitKVPipelineTag::QR_NWARP_SSHUFFLE,
                           BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS<PipelineProblem>,
                           BlockFmhaFwdSplitKVPipelineQRKSVS<PipelineProblem>>;

    using Epilogue = Default2DEpilogue<Default2DEpilogueProblem<float, float, false, false>>;

    using Kernel = FmhaFwdSplitKVKernel<Pipeline, Epilogue>;

    // Tensor layouts:
    // Q: [batch, nhead, M, K]
    // K: [batch, nhead_k, N, K]
    // V: [batch, nhead_k, N, O] (rowmajor) or [batch, nhead_k, O, N] (colmajor)
    // o_acc: [batch, nhead, num_splits, M, O] (workspace output)
    // lse_acc: [batch, nhead, num_splits, M] (workspace output)
    struct Descriptor
    {
        index_t batch, nhead, nhead_k, M, K;
        index_t q_stride_batch, q_stride_nhead, q_stride_m;

        index_t N;
        index_t k_stride_batch, k_stride_nhead, k_stride_n;

        index_t O;
        index_t v_stride_batch, v_stride_nhead, v_stride_n;

        index_t num_splits;

        index_t o_acc_stride_batch, o_acc_stride_nhead, o_acc_stride_split, o_acc_stride_m;
        index_t lse_acc_stride_batch, lse_acc_stride_nhead, lse_acc_stride_split;

        CK_TILE_HOST_DEVICE constexpr bool IsValid() const { return true; }
    };

    template <typename QDims,
              typename QStrides,
              typename KDims,
              typename KStrides,
              typename VDims,
              typename VStrides,
              typename OAccDims,
              typename OAccStrides,
              typename LseAccStrides>
    CK_TILE_HOST_DEVICE static constexpr auto make_descriptor(QDims q_dims,
                                                              QStrides q_strides,
                                                              KDims k_dims,
                                                              KStrides k_strides,
                                                              VDims v_dims,
                                                              VStrides v_strides,
                                                              OAccDims o_acc_dims,
                                                              OAccStrides o_acc_strides,
                                                              LseAccStrides lse_acc_strides)
    {
        return Descriptor{q_dims[number<0>{}],
                          q_dims[number<1>{}],
                          k_dims[number<1>{}], // nhead_k from K tensor
                          q_dims[number<2>{}],
                          q_dims[number<3>{}],
                          q_strides[number<0>{}],
                          q_strides[number<1>{}],
                          q_strides[number<2>{}],
                          //
                          k_dims[number<2>{}],
                          k_strides[number<0>{}],
                          k_strides[number<1>{}],
                          k_strides[number<2>{}],
                          //
                          v_dims[number<3>{}],
                          v_strides[number<0>{}],
                          v_strides[number<1>{}],
                          v_strides[number<2>{}],
                          //
                          o_acc_dims[number<2>{}], // num_splits
                          o_acc_strides[number<0>{}],
                          o_acc_strides[number<1>{}],
                          o_acc_strides[number<2>{}],
                          o_acc_strides[number<3>{}],
                          //
                          lse_acc_strides[number<0>{}],
                          lse_acc_strides[number<1>{}],
                          lse_acc_strides[number<2>{}]};
    }

    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   float scale_s,
                                   const DataType_* q_ptr,
                                   const DataType_* k_ptr,
                                   const DataType_* v_ptr,
                                   float* lse_acc_ptr,
                                   float* o_acc_ptr)
    {
        using Kargs = typename Kernel::Kargs;
        Kargs kargs{};

        kargs.q_ptr       = q_ptr;
        kargs.k_ptr       = k_ptr;
        kargs.v_ptr       = v_ptr;
        kargs.lse_acc_ptr = lse_acc_ptr;
        kargs.o_acc_ptr   = o_acc_ptr;
        kargs.sink_ptr    = nullptr;

        kargs.batch    = desc.batch;
        kargs.seqlen_q = desc.M;
        kargs.seqlen_k = desc.N;
        kargs.hdim_q   = desc.K;
        kargs.hdim_v   = desc.O;

        kargs.num_head_q     = desc.nhead;
        kargs.nhead_ratio_qk = desc.nhead / desc.nhead_k;
        kargs.num_splits     = desc.num_splits;

        kargs.scale_s = scale_s;

        kargs.stride_q     = desc.q_stride_m;
        kargs.stride_k     = desc.k_stride_n;
        kargs.stride_v     = desc.v_stride_n;
        kargs.stride_o_acc = desc.o_acc_stride_m;

        kargs.nhead_stride_q       = desc.q_stride_nhead;
        kargs.nhead_stride_k       = desc.k_stride_nhead;
        kargs.nhead_stride_v       = desc.v_stride_nhead;
        kargs.nhead_stride_lse_acc = desc.lse_acc_stride_nhead;
        kargs.nhead_stride_o_acc   = desc.o_acc_stride_nhead;

        kargs.split_stride_lse_acc = desc.lse_acc_stride_split;
        kargs.split_stride_o_acc   = desc.o_acc_stride_split;

        kargs.seqlen_k_ptr    = nullptr;
        kargs.cache_batch_idx = nullptr;

        kargs.batch_stride_q       = desc.q_stride_batch;
        kargs.batch_stride_k       = desc.k_stride_batch;
        kargs.batch_stride_v       = desc.v_stride_batch;
        kargs.batch_stride_lse_acc = desc.lse_acc_stride_batch;
        kargs.batch_stride_o_acc   = desc.o_acc_stride_batch;

        Kernel{}(kargs);
    }
};

} // namespace ck_tile
