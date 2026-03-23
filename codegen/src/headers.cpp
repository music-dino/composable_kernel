// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/headers.hpp"
#include "ck_headers.hpp"
#include "ck_tile_headers.hpp"
#include "ck_codegen_headers.hpp"

#include <cctype>
#include <cstring>

namespace ck {
namespace host {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
const std::string config_header = "";
#pragma clang diagnostic pop

static constexpr const char* HOST_TOKEN   = "CK_TILE_HOST";
static constexpr size_t HOST_TOKEN_LEN    = 12;
static constexpr const char* REPLACEMENT               = "{ __builtin_unreachable(); }";
static constexpr size_t REPLACEMENT_LEN                = 28;
static constexpr const char* CONSTEXPR_REPLACEMENT     = "{ return {}; }";
static constexpr size_t CONSTEXPR_REPLACEMENT_LEN      = 14;
static constexpr const char* CONSTEXPR_AUTO_REPLACEMENT = "{ return 0; }";
static constexpr size_t CONSTEXPR_AUTO_REPLACEMENT_LEN  = 13;

enum class ScanState
{
    Normal,
    InString,
    InChar,
    InLineComment,
    InBlockComment,
    InRawString
};

// Find the matching closing brace for an opening brace at position `open`.
// Returns the position of the matching '}', or std::string::npos if not found.
static size_t find_matching_brace(const std::string& s, size_t open)
{
    int depth        = 1;
    size_t i         = open + 1;
    size_t len       = s.size();
    ScanState state  = ScanState::Normal;
    std::string raw_delim;

    while(i < len && depth > 0)
    {
        char c = s[i];

        switch(state)
        {
        case ScanState::Normal:
            if(c == '{')
                ++depth;
            else if(c == '}')
                --depth;
            else if(c == '"')
            {
                // Check for raw string literal: R"delim(
                if(i > 0 && s[i - 1] == 'R')
                {
                    state = ScanState::InRawString;
                    raw_delim.clear();
                    ++i;
                    while(i < len && s[i] != '(')
                    {
                        raw_delim += s[i];
                        ++i;
                    }
                }
                else
                {
                    state = ScanState::InString;
                }
            }
            else if(c == '\'')
                state = ScanState::InChar;
            else if(c == '/' && i + 1 < len)
            {
                if(s[i + 1] == '/')
                {
                    state = ScanState::InLineComment;
                    ++i;
                }
                else if(s[i + 1] == '*')
                {
                    state = ScanState::InBlockComment;
                    ++i;
                }
            }
            break;

        case ScanState::InString:
            if(c == '\\')
                ++i; // skip escaped character
            else if(c == '"')
                state = ScanState::Normal;
            break;

        case ScanState::InChar:
            if(c == '\\')
                ++i;
            else if(c == '\'')
                state = ScanState::Normal;
            break;

        case ScanState::InLineComment:
            if(c == '\n')
                state = ScanState::Normal;
            break;

        case ScanState::InBlockComment:
            if(c == '*' && i + 1 < len && s[i + 1] == '/')
            {
                state = ScanState::Normal;
                ++i;
            }
            break;

        case ScanState::InRawString:
        {
            // Look for )delim"
            if(c == ')' && i + raw_delim.size() + 1 < len &&
               s.compare(i + 1, raw_delim.size(), raw_delim) == 0 &&
               s[i + 1 + raw_delim.size()] == '"')
            {
                i += raw_delim.size() + 1;
                state = ScanState::Normal;
            }
            break;
        }
        }
        ++i;
    }

    return (depth == 0) ? i - 1 : std::string::npos;
}

std::string strip_host_bodies(const std::string& content)
{
    std::string result;
    result.reserve(content.size());
    size_t pos = 0;
    size_t len = content.size();

    while(pos < len)
    {
        size_t found = content.find(HOST_TOKEN, pos);
        if(found == std::string::npos)
        {
            result.append(content, pos, len - pos);
            break;
        }

        // Check it's not CK_TILE_HOST_DEVICE (followed by '_')
        size_t after_token = found + HOST_TOKEN_LEN;
        if(after_token < len && content[after_token] == '_')
        {
            result.append(content, pos, after_token - pos);
            pos = after_token;
            continue;
        }

        // Also verify it's not a substring of a longer identifier (preceded by alnum or _)
        if(found > 0 && (std::isalnum(content[found - 1]) || content[found - 1] == '_'))
        {
            result.append(content, pos, after_token - pos);
            pos = after_token;
            continue;
        }

        // Skip if this is on a #define line
        {
            size_t line_start = content.rfind('\n', found);
            line_start        = (line_start == std::string::npos) ? 0 : line_start + 1;
            auto line_prefix  = content.substr(line_start, found - line_start);
            if(line_prefix.find("#define") != std::string::npos)
            {
                result.append(content, pos, after_token - pos);
                pos = after_token;
                continue;
            }
        }

        // Found a standalone CK_TILE_HOST. Copy everything up to after the token.
        result.append(content, pos, after_token - pos);
        pos = after_token;

        // Scan forward to find '{' or ';', skipping strings/comments/parens.
        // We need to handle constructor initializer lists (: ... { }) too.
        int paren_depth     = 0;
        bool is_constexpr   = false;
        bool is_auto_return = false;
        ScanState state     = ScanState::Normal;
        size_t scan         = pos;
        size_t body_start   = std::string::npos;
        bool found_body     = false;
        std::string raw_delim;

        while(scan < len)
        {
            char c = content[scan];

            switch(state)
            {
            case ScanState::Normal:
                if(c == 'c' && !is_constexpr && scan + 9 <= len &&
                   content.compare(scan, 9, "constexpr") == 0 &&
                   (scan + 9 >= len || !std::isalnum(content[scan + 9])))
                {
                    is_constexpr = true;
                }

                if(c == 'a' && paren_depth == 0 && !is_auto_return &&
                   scan + 4 <= len &&
                   content.compare(scan, 4, "auto") == 0 &&
                   (scan + 4 >= len ||
                    (!std::isalnum(content[scan + 4]) && content[scan + 4] != '_')))
                {
                    is_auto_return = true;
                }

                if(c == '(')
                    ++paren_depth;
                else if(c == ')')
                    --paren_depth;
                else if(paren_depth == 0)
                {
                    if(c == ';')
                    {
                        goto done_scanning;
                    }
                    else if(c == '{')
                    {
                        body_start = scan;
                        found_body = true;
                        goto done_scanning;
                    }
                }

                if(c == '"')
                {
                    if(scan > 0 && content[scan - 1] == 'R')
                    {
                        state = ScanState::InRawString;
                        raw_delim.clear();
                        ++scan;
                        while(scan < len && content[scan] != '(')
                        {
                            raw_delim += content[scan];
                            ++scan;
                        }
                    }
                    else
                    {
                        state = ScanState::InString;
                    }
                }
                else if(c == '\'')
                    state = ScanState::InChar;
                else if(c == '/' && scan + 1 < len)
                {
                    if(content[scan + 1] == '/')
                    {
                        state = ScanState::InLineComment;
                        ++scan;
                    }
                    else if(content[scan + 1] == '*')
                    {
                        state = ScanState::InBlockComment;
                        ++scan;
                    }
                }
                break;

            case ScanState::InString:
                if(c == '\\')
                    ++scan;
                else if(c == '"')
                    state = ScanState::Normal;
                break;

            case ScanState::InChar:
                if(c == '\\')
                    ++scan;
                else if(c == '\'')
                    state = ScanState::Normal;
                break;

            case ScanState::InLineComment:
                if(c == '\n')
                    state = ScanState::Normal;
                break;

            case ScanState::InBlockComment:
                if(c == '*' && scan + 1 < len && content[scan + 1] == '/')
                {
                    state = ScanState::Normal;
                    ++scan;
                }
                break;

            case ScanState::InRawString:
                if(c == ')' && scan + raw_delim.size() + 1 < len &&
                   content.compare(scan + 1, raw_delim.size(), raw_delim) == 0 &&
                   content[scan + 1 + raw_delim.size()] == '"')
                {
                    scan += raw_delim.size() + 1;
                    state = ScanState::Normal;
                }
                break;
            }
            ++scan;
        }

    done_scanning:
        if(!found_body)
        {
            // No body found (declaration or EOF). Copy everything scanned.
            result.append(content, pos, scan - pos + 1);
            pos = scan + 1;
            continue;
        }

        // Copy from current pos to just before the opening brace (preserves
        // the function signature and any constructor initializer list)
        result.append(content, pos, body_start - pos);

        // Find matching closing brace
        size_t body_end = find_matching_brace(content, body_start);
        if(body_end == std::string::npos)
        {
            // Malformed -- just copy the rest unchanged
            result.append(content, body_start, len - body_start);
            pos = len;
            continue;
        }

        if(is_constexpr && is_auto_return)
            result.append(CONSTEXPR_AUTO_REPLACEMENT, CONSTEXPR_AUTO_REPLACEMENT_LEN);
        else if(is_constexpr)
            result.append(CONSTEXPR_REPLACEMENT, CONSTEXPR_REPLACEMENT_LEN);
        else
            result.append(REPLACEMENT, REPLACEMENT_LEN);
        pos = body_end + 1;
    }

    return result;
}

std::unordered_map<std::string_view, std::string_view> GetHeaders()
{
    auto headers = ck_headers();
    headers.insert(std::make_pair("ck/config.h", config_header));
    return headers;
}

std::unordered_map<std::string_view, std::string_view> GetTileHeaders()
{
    auto headers = ck_tile_headers();
    auto codegen_hdrs = ck_codegen_headers();
    headers.insert(codegen_hdrs.begin(), codegen_hdrs.end());
    return headers;
}

std::unordered_map<std::string, std::string> GetTileHeadersForRTC()
{
    auto tile_hdrs    = ck_tile_headers();
    auto codegen_hdrs = ck_codegen_headers();

    std::unordered_map<std::string, std::string> result;
    result.reserve(tile_hdrs.size() + codegen_hdrs.size());

    for(auto& [name, content] : tile_hdrs)
        result.emplace(std::string(name), strip_host_bodies(std::string(content)));

    for(auto& [name, content] : codegen_hdrs)
        result.emplace(std::string(name), strip_host_bodies(std::string(content)));

    return result;
}

} // namespace host
} // namespace ck
