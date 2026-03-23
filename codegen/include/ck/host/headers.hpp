// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace ck {
namespace host {

std::unordered_map<std::string_view, std::string_view> GetHeaders();

std::unordered_map<std::string_view, std::string_view> GetTileHeaders();

std::unordered_map<std::string, std::string> GetTileHeadersForRTC();

std::string strip_host_bodies(const std::string& content);

} // namespace host
} // namespace ck
