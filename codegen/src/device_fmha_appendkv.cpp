// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_appendkv/problem.hpp"
#include "ck/host/device_fmha_appendkv/operation.hpp"

namespace ck {
namespace host {
namespace device_fmha_appendkv {

std::string Problem::GetIncludeHeader() const { return "ck/host/device_fmha_appendkv/wrapper.hpp"; }

std::vector<Solution> Problem::GetSolutions(const std::string& arch) const
{
    std::vector<Solution> solutions;
    auto operations = Operation::CreateOperations(*this, arch);
    for(const auto& op : operations)
    {
        solutions.push_back(op.ToSolution());
    }
    return solutions;
}

} // namespace device_fmha_appendkv
} // namespace host
} // namespace ck
