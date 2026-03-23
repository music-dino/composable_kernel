# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)
project(composable_kernel_host)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
configure_file(${CK_ROOT}/include/ck/config.h.in ${CK_ROOT}/include/ck/config.h)

find_package(ROCM)
include(ROCMInstallTargets)
include(ROCMTest)

rocm_setup_version(VERSION 1.0)

list(APPEND CMAKE_MODULE_PATH ${CK_ROOT}/cmake)
include(Embed)
file(GLOB_RECURSE KERNEL_FILES CONFIGURE_DEPENDS
    ${CK_ROOT}/include/ck/*.hpp)
add_embed_library(ck_headers ${KERNEL_FILES} RELATIVE ${CK_ROOT}/include)

# Embed CK Tile headers (ck_tile/*.hpp) for FMHA RTC API
file(GLOB_RECURSE CK_TILE_KERNEL_FILES CONFIGURE_DEPENDS
    ${CK_ROOT}/include/ck_tile/*.hpp)
add_embed_library(ck_tile_headers ${CK_TILE_KERNEL_FILES} RELATIVE ${CK_ROOT}/include)

# Embed codegen device headers (wrapper.hpp for FMHA RTC)
file(GLOB_RECURSE CK_CODEGEN_DEVICE_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ck/host/device_fmha_fwd/wrapper.hpp)
add_embed_library(ck_codegen_headers ${CK_CODEGEN_DEVICE_FILES} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Embed rocm-cxx replacement headers for HIPRTC
file(GLOB_RECURSE ROCM_CXX_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rocm-cxx/rocm/*.hpp)
add_embed_library(ck_rocm_headers ${ROCM_CXX_FILES}
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include/rocm-cxx)

# Generate shim standard headers that redirect to rocm/ implementations
set(ROCM_SHIM_OVERRIDES "stddef:cstddef;stdint:cstdint;assert:cassert")
set(ROCM_SHIM_SKIP "config;integral_constant;stdint;stddef")
set(SHIM_DIR ${CMAKE_CURRENT_BINARY_DIR}/rocm_shims)
file(MAKE_DIRECTORY ${SHIM_DIR})

file(GLOB ROCM_TOP_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rocm-cxx/rocm/*.hpp)

set(SHIM_FILES "")
foreach(HEADER ${ROCM_TOP_HEADERS})
    get_filename_component(ROCM_NAME ${HEADER} NAME_WE)
    list(FIND ROCM_SHIM_SKIP ${ROCM_NAME} SKIP_IDX)
    if(NOT SKIP_IDX EQUAL -1)
        continue()
    endif()
    set(STD_NAME ${ROCM_NAME})
    foreach(OVERRIDE ${ROCM_SHIM_OVERRIDES})
        string(REPLACE ":" ";" PARTS ${OVERRIDE})
        list(GET PARTS 0 FROM)
        list(GET PARTS 1 TO)
        if(ROCM_NAME STREQUAL FROM)
            set(STD_NAME ${TO})
        endif()
    endforeach()
    file(WRITE "${SHIM_DIR}/${STD_NAME}"
        "#pragma once\n#include <rocm/${ROCM_NAME}.hpp>\nnamespace std { using namespace rocm; }\n")
    list(APPEND SHIM_FILES "${SHIM_DIR}/${STD_NAME}")
endforeach()

# Copy hand-written shims (cmath, cstdint, stdint.h, cstddef) into the shim directory
file(GLOB MANUAL_SHIM_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rocm-cxx/shims/*)
foreach(MANUAL_SHIM ${MANUAL_SHIM_FILES})
    get_filename_component(SHIM_NAME ${MANUAL_SHIM} NAME)
    configure_file(${MANUAL_SHIM} "${SHIM_DIR}/${SHIM_NAME}" COPYONLY)
    list(APPEND SHIM_FILES "${SHIM_DIR}/${SHIM_NAME}")
endforeach()

add_embed_library(ck_rocm_shim_headers ${SHIM_FILES}
    RELATIVE ${SHIM_DIR})

add_compile_options(-std=c++20)

file(GLOB SOURCES CONFIGURE_DEPENDS src/*.cpp)
# TODO: Use object library
add_library(ck_host STATIC ${SOURCES})
target_link_libraries(ck_host PRIVATE ck_headers ck_tile_headers ck_codegen_headers
    ck_rocm_headers ck_rocm_shim_headers)

set_target_properties(ck_host PROPERTIES 
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON)

# target_include_directories(ck_host PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
# )

add_executable(ck-template-driver driver/main.cpp)
target_link_libraries(ck-template-driver ck_host)

rocm_install_targets(
    TARGETS ck_host ck_headers ck_tile_headers ck_codegen_headers
        ck_rocm_headers ck_rocm_shim_headers
    EXPORT ck_host_targets
    INCLUDE include
)
rocm_export_targets(
    TARGETS ck_host ck_headers ck_tile_headers ck_codegen_headers
        ck_rocm_headers ck_rocm_shim_headers
    EXPORT ck_host_targets
    NAMESPACE composable_kernel::
)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()

