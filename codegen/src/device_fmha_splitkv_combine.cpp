// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_splitkv_combine/problem.hpp"
#include "ck/host/device_fmha_splitkv_combine/operation.hpp"
#include <algorithm>
#include <vector>

namespace ck {
namespace host {
namespace device_fmha_splitkv_combine {

std::string Problem::GetIncludeHeader() const
{
    return "ck/host/device_fmha_splitkv_combine/wrapper.hpp";
}

std::vector<Solution> Problem::GetSolutions(const std::string& arch) const
{
    if(!IsSupportedArch(arch))
        return {};

    auto ops = Operation::CreateOperations(*this, arch);
    std::vector<Solution> result;
    std::transform(ops.begin(), ops.end(), std::back_inserter(result), [](const auto& op) {
        return op.ToSolution();
    });
    return result;
}

} // namespace device_fmha_splitkv_combine
} // namespace host
} // namespace ck
