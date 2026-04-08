// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

namespace ck_tile {

template <typename DataType,
          ck_tile::index_t kM0,
          ck_tile::index_t kN0,
          ck_tile::index_t kK0,
          ck_tile::index_t kN1,
          bool kIsVLayoutRowMajor,
          bool kPadSeqLenQ,
          bool kPadSeqLenK,
          bool kPadHeadDimQ,
          bool kPadHeadDimV,
          ck_tile::RotaryEmbeddingEnum kRotaryEnum>
struct FmhaFwdAppendKVWrapper
{
    using FmhaTraits = ck_tile::
        TileFmhaFwdAppendKVTraits<kPadSeqLenQ, kPadSeqLenK, kPadHeadDimQ, kPadHeadDimV, -1>;

    using FmhaPipelineProblem = ck_tile::BlockFmhaFwdAppendKVPipelineProblem<DataType,
                                                                             DataType,
                                                                             DataType,
                                                                             kM0,
                                                                             kN0,
                                                                             kK0,
                                                                             kN1,
                                                                             kIsVLayoutRowMajor,
                                                                             kRotaryEnum,
                                                                             false, // kIsPagedKV
                                                                             FmhaTraits>;

    using FmhaPipeline = ck_tile::BlockFmhaFwdAppendKVPipeline<FmhaPipelineProblem>;

    using Kernel = ck_tile::FmhaFwdAppendKVKernel<FmhaPipeline>;

    struct Descriptor
    {
        ck_tile::index_t batch;
        ck_tile::index_t nhead;
        ck_tile::index_t nhead_k;
        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_knew;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        // Q strides [batch, nhead, seqlen_q, hdim_q]
        ck_tile::index_t stride_q;
        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t batch_stride_q;

        // K strides [batch, nhead_k, seqlen_k, hdim_q]
        ck_tile::index_t stride_k;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t batch_stride_k;

        // Knew strides [batch, nhead_k, seqlen_knew, hdim_q]
        ck_tile::index_t stride_knew;
        ck_tile::index_t nhead_stride_knew;
        ck_tile::index_t batch_stride_knew;

        // V strides [batch, nhead_k, seqlen_k, hdim_v]
        ck_tile::index_t stride_v;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t batch_stride_v;

        // Vnew strides [batch, nhead_k, seqlen_knew, hdim_v]
        ck_tile::index_t stride_vnew;
        ck_tile::index_t nhead_stride_vnew;
        ck_tile::index_t batch_stride_vnew;

        // Rotary embedding (optional)
        ck_tile::index_t rotary_dim;
        bool has_mask;

        CK_TILE_DEVICE constexpr bool IsValid() const { return true; }
    };

    template <typename QDims,
              typename QStrides,
              typename KDims,
              typename KStrides,
              typename KnewDims,
              typename KnewStrides,
              typename VDims,
              typename VStrides,
              typename VnewDims,
              typename VnewStrides>
    CK_TILE_DEVICE static constexpr Descriptor make_descriptor(QDims q_dims,
                                                               QStrides q_strides,
                                                               KDims k_dims,
                                                               KStrides k_strides,
                                                               KnewDims knew_dims,
                                                               KnewStrides knew_strides,
                                                               VDims v_dims,
                                                               VStrides v_strides,
                                                               VnewDims vnew_dims,
                                                               VnewStrides vnew_strides,
                                                               ck_tile::index_t rotary_dim = 0,
                                                               bool has_mask               = false)
    {
        // Q: [batch, nhead, seqlen_q, hdim_q]
        // K: [batch, nhead_k, seqlen_k, hdim_q]
        // Knew: [batch, nhead_k, seqlen_knew, hdim_q]
        // V: [batch, nhead_k, seqlen_k, hdim_v]
        // Vnew: [batch, nhead_k, seqlen_knew, hdim_v]

        return Descriptor{
            q_dims[number<0>{}],    // batch
            q_dims[number<1>{}],    // nhead
            k_dims[number<1>{}],    // nhead_k
            q_dims[number<2>{}],    // seqlen_q
            knew_dims[number<2>{}], // seqlen_knew
            q_dims[number<3>{}],    // hdim_q
            v_dims[number<3>{}],    // hdim_v

            q_strides[number<2>{}], // stride_q
            q_strides[number<1>{}], // nhead_stride_q
            q_strides[number<0>{}], // batch_stride_q

            k_strides[number<2>{}], // stride_k
            k_strides[number<1>{}], // nhead_stride_k
            k_strides[number<0>{}], // batch_stride_k

            knew_strides[number<2>{}], // stride_knew
            knew_strides[number<1>{}], // nhead_stride_knew
            knew_strides[number<0>{}], // batch_stride_knew

            v_strides[number<2>{}], // stride_v
            v_strides[number<1>{}], // nhead_stride_v
            v_strides[number<0>{}], // batch_stride_v

            vnew_strides[number<2>{}], // stride_vnew
            vnew_strides[number<1>{}], // nhead_stride_vnew
            vnew_strides[number<0>{}], // batch_stride_vnew

            rotary_dim,
            has_mask,
        };
    }

    CK_TILE_DEVICE static void Run(const Descriptor& desc,
                                   DataType* q_ptr,
                                   DataType* k_ptr,
                                   const DataType* knew_ptr,
                                   DataType* v_ptr,
                                   const DataType* vnew_ptr,
                                   const int32_t* seqlen_k_ptr,
                                   const DataType* rotary_cos_ptr,
                                   const DataType* rotary_sin_ptr,
                                   const int32_t* cache_batch_idx)
    {
        using Kargs = typename Kernel::Kargs;
        Kargs kargs{};

        kargs.q_ptr    = q_ptr;
        kargs.k_ptr    = k_ptr;
        kargs.knew_ptr = knew_ptr;
        kargs.v_ptr    = v_ptr;
        kargs.vnew_ptr = vnew_ptr;

        kargs.seqlen_k_ptr = seqlen_k_ptr;

        kargs.seqlen_q    = desc.seqlen_q;
        kargs.seqlen_knew = desc.seqlen_knew;
        kargs.hdim_q      = desc.hdim_q;
        kargs.hdim_v      = desc.hdim_v;

        kargs.num_head_q     = desc.nhead;
        kargs.nhead_ratio_qk = desc.nhead / desc.nhead_k;

        kargs.stride_q    = desc.stride_q;
        kargs.stride_k    = desc.stride_k;
        kargs.stride_knew = desc.stride_knew;
        kargs.stride_v    = desc.stride_v;
        kargs.stride_vnew = desc.stride_vnew;

        kargs.nhead_stride_q    = desc.nhead_stride_q;
        kargs.nhead_stride_k    = desc.nhead_stride_k;
        kargs.nhead_stride_knew = desc.nhead_stride_knew;
        kargs.nhead_stride_v    = desc.nhead_stride_v;
        kargs.nhead_stride_vnew = desc.nhead_stride_vnew;

        kargs.batch_stride_q    = desc.batch_stride_q;
        kargs.batch_stride_k    = desc.batch_stride_k;
        kargs.batch_stride_knew = desc.batch_stride_knew;
        kargs.batch_stride_v    = desc.batch_stride_v;
        kargs.batch_stride_vnew = desc.batch_stride_vnew;

        if constexpr(kRotaryEnum != ck_tile::RotaryEmbeddingEnum::NONE)
        {
            kargs.rotary_cos_ptr = rotary_cos_ptr;
            kargs.rotary_sin_ptr = rotary_sin_ptr;
            kargs.rotary_dim     = desc.rotary_dim;
            kargs.has_mask       = desc.has_mask;
        }

        kargs.cache_batch_idx = cache_batch_idx;

        Kernel{}(kargs);
    }
};

} // namespace ck_tile
