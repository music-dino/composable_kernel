# SplitKV RTC API

This document describes how to use the SplitKV Real-Time Compilation (RTC) API for Flash Attention decode workloads. The SplitKV kernel splits the K/V sequence into chunks, computes partial attention outputs for each chunk, and writes intermediate results (`o_acc` and `lse_acc`) to workspace buffers. These are later consumed by a separate Combine kernel to produce the final output.

## Headers

```cpp
#include "ck/host/device_fmha_splitkv/problem.hpp"
#include "ck/host/device_fmha_splitkv/operation.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "ck/host/headers.hpp"
#include <rtc/compile_kernel.hpp>
```

The common test helpers (kernel template string, params struct, launch dims) are in:

```cpp
#include "fmha_fwd_splitkv_common.hpp"  // codegen/test/include/
```

## Namespace

```cpp
namespace splitkv = ck::host::device_fmha_splitkv;
```

## Step 1: Define the Problem

```cpp
splitkv::Problem prob;
prob.batch      = 2;
prob.nhead      = 8;      // number of Q heads
prob.nhead_k    = 4;      // number of K/V heads (nhead_k <= nhead for GQA/MQA)
prob.M          = 1;      // seqlen_q (typically 1 for decode)
prob.N          = 512;    // seqlen_k (full KV cache length)
prob.K          = 128;    // hdim_q
prob.O          = 128;    // hdim_v
prob.num_splits = 4;      // number of KV splits
prob.dtype      = ck::host::DataType::Half;
prob.o_acc_dtype = ck::host::DataType::Float;  // workspace output type (default: Float)
prob.is_v_rowmajor = true;  // V layout: true=[N,O], false=[O,N]
```

### Problem fields

| Field | Description |
|-------|-------------|
| `M` | Query sequence length. Typically 1 for decode, but can be > 1. |
| `N` | Key/Value sequence length (full KV cache length). |
| `K` | Head dimension for Q and K. |
| `O` | Head dimension for V (can differ from K). |
| `batch` | Batch size. |
| `nhead` | Number of Q attention heads. |
| `nhead_k` | Number of K/V attention heads. Must divide `nhead`. |
| `num_splits` | Number of chunks to split the K/V sequence into. |
| `dtype` | Input data type (`DataType::Half`). |
| `o_acc_dtype` | Output accumulation type for the `o_acc` workspace. Default `DataType::Float`. Must be `Float` if the Combine kernel will consume the output (the Combine kernel requires `float` for `o_acc` due to an internal constraint tying LSE type to `o_acc` type). |
| `is_v_rowmajor` | V tensor layout. `true` = `[batch, nhead_k, N, O]`, `false` = `[batch, nhead_k, O, N]`. |

## Step 2: Get Solutions

```cpp
auto solutions = prob.GetSolutions("gfx90a");  // or "gfx942", etc.
```

This returns a `std::vector<ck::host::Solution>`. Each solution represents a different tile configuration and pipeline variant. Multiple solutions are returned; the caller can try them all and pick the best-performing one.

### Inspecting a solution

```cpp
// Get the full template instantiation string
std::string tmpl = solution.ToTemplateString();

// Query individual template parameters
auto bm0 = solution.GetTemplateParameter<std::size_t>("BM0");
auto pipeline = solution.GetTemplateParameter<std::string>("PipelineTag");
```

Available template parameters: `DataType`, `OaccOutType`, `BM0`, `BN0`, `BK0`, `BN1`, `BK1`, `BK0Max`, `RM0`, `RN0`, `RK0`, `RM1`, `RN1`, `RK1`, `WM0`, `WN0`, `WK0`, `WM1`, `WN1`, `WK1`, `IsVRowMajor`, `PadM`, `PadN`, `PadK`, `PadO`, `HasUnevenSplits`, `MergeNumHeadGroupsSeqLenQ`, `PipelineTag`.

## Step 3: Compute Parameters and Strides

```cpp
float scale_s = 1.0f / std::sqrt(static_cast<float>(prob.K));
ck::host::SplitKVParams params = ck::host::make_splitkv_params(prob, scale_s);
```

`make_splitkv_params` computes contiguous strides for all tensors:

### Tensor layouts and strides

| Tensor | Shape | Strides (innermost-last) |
|--------|-------|--------------------------|
| Q | `[batch, nhead, M, K]` | `q_stride_batch`, `q_stride_nhead`, `q_stride_m` (innermost K is implicit stride 1) |
| K | `[batch, nhead_k, N, K]` | `k_stride_batch`, `k_stride_nhead`, `k_stride_n` |
| V | `[batch, nhead_k, N, O]` | `v_stride_batch`, `v_stride_nhead`, `v_stride_n` |
| o_acc | `[batch, nhead, num_splits, M, O]` | `o_acc_stride_batch`, `o_acc_stride_nhead`, `o_acc_stride_split`, `o_acc_stride_m` |
| lse_acc | `[batch, nhead, num_splits, M]` | `lse_acc_stride_batch`, `lse_acc_stride_nhead`, `lse_acc_stride_split` |

## Step 4: Generate Kernel Source

```cpp
std::string source = ck::host::make_splitkv_kernel_source(prob, solution, params);
```

This interpolates the kernel template with the problem dimensions, strides, and the solution's template string. The generated kernel has the signature:

```cpp
extern "C" __global__ void f(const fp16_t* q, const fp16_t* k, const fp16_t* v,
                              float* o_acc, float* lse_acc);
```

## Step 5: Compile the Kernel

```cpp
auto srcs = get_tile_headers_for_test();  // embedded CK tile headers
srcs.push_back({"main.cpp", source});

rtc::compile_options opts;
opts.kernel_name = "f";

auto kernel = rtc::compile_kernel(srcs, opts);
```

`get_tile_headers_for_test()` returns the embedded CK tile headers required for compilation.

## Step 6: Compute Launch Dimensions

```cpp
auto [grid, block] = ck::host::get_splitkv_launch_dims(solution, prob);
```

The grid and block dimensions are derived from the solution's tile parameters:

- **grid.x** = `ceil(M_eff / BM0) * ceil(O / BN1) * num_splits`
- **grid.y** = `nhead_eff`
- **grid.z** = `batch`
- **block.x** = `num_warps * 64` (warp size = 64 on AMD GCN/CDNA)

When `MergeNumHeadGroupsSeqLenQ` is true (GQA optimization for decode with `seqlen_q=1` and `nhead_k < nhead`):
- `M_eff = M * (nhead / nhead_k)`
- `nhead_eff = nhead_k`

## Step 7: Launch

The compiled kernel has the following signature:

```cpp
void f(const fp16_t* q, const fp16_t* k, const fp16_t* v, float* o_acc, float* lse_acc);
```

Launch it with the grid and block dimensions from Step 6.

### Buffer initialization requirements

- `o_acc` must be initialized to `0.0f` before launch
- `lse_acc` must be initialized to `-infinity` before launch

## Complete Example

```cpp
#include "ck/host/device_fmha_splitkv/problem.hpp"
#include "ck/host/device_fmha_splitkv/operation.hpp"
#include "ck/host/utils.hpp"
#include "ck/host/headers.hpp"
#include "fmha_fwd_splitkv_common.hpp"
#include <rtc/compile_kernel.hpp>

namespace splitkv = ck::host::device_fmha_splitkv;

void run_splitkv()
{
    // 1. Define the problem
    splitkv::Problem prob;
    prob.batch      = 2;
    prob.nhead      = 8;
    prob.nhead_k    = 4;
    prob.M          = 1;
    prob.N          = 512;
    prob.K          = 128;
    prob.O          = 128;
    prob.num_splits = 4;
    prob.dtype      = ck::host::DataType::Half;

    // 2. Get solutions and pick one
    auto solutions = prob.GetSolutions("gfx90a");
    auto& solution = solutions[0];

    // 3. Compute params (dimensions + strides)
    float scale_s = 1.0f / std::sqrt(static_cast<float>(prob.K));
    auto params = ck::host::make_splitkv_params(prob, scale_s);

    // 4. Generate kernel source
    auto source = ck::host::make_splitkv_kernel_source(prob, solution, params);

    // 5. Compile
    auto srcs = get_tile_headers_for_test();
    srcs.push_back({"main.cpp", source});
    rtc::compile_options opts;
    opts.kernel_name = "f";
    auto kernel = rtc::compile_kernel(srcs, opts);

    // 6. Get launch dimensions
    auto [grid, block] = ck::host::get_splitkv_launch_dims(solution, prob);

    // 7. Launch (user provides device pointers for q, k, v, o_acc, lse_acc)
    kernel.launch(nullptr, grid, block)(q_ptr, k_ptr, v_ptr, o_acc_ptr, lse_acc_ptr);
}
```

## Output

After the kernel completes:

- `o_acc` contains partial attention outputs: `[batch, nhead, num_splits, M, O]` in `float`
- `lse_acc` contains log-sum-exp values per split: `[batch, nhead, num_splits, M]` in `float`

These are intermediate results. To produce the final attention output, pass them to the SplitKV Combine kernel, which weights each split's output by `exp(local_LSE - global_LSE)` and sums them.

## Notes

- The kernel supports GQA (Grouped Query Attention) and MQA (Multi-Query Attention) via `nhead_k < nhead`.
- The `num_splits` does not need to evenly divide `N`; the kernel handles uneven splits internally.
- The `o_acc_dtype` field in `Problem` controls the output type of the `o_acc` workspace. It defaults to `Float`. If the Combine kernel will consume the output, it must remain `Float` due to an internal kernel constraint (`static_assert(LSEDataType == OaccDataType)` in the Combine kernel pipeline).
- Supported architectures: GFX9 family (e.g., `gfx90a`, `gfx942`) except GFX950.
