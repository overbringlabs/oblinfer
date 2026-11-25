# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
cmake_minimum_required(VERSION 3.28)

# ingine is compiled only for sm_120a (RTX 5090 class). Set and validate the architecture before
# CUDA compiler detection.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "ingine supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

# Prefer the installer's private toolkit, then /opt/cuda, when nvcc is not chosen explicitly.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  foreach(_nvcc ${CMAKE_CURRENT_SOURCE_DIR}/../.cuda-toolkit/nvidia/cu13/bin/nvcc
                ${CMAKE_CURRENT_SOURCE_DIR}/../.cuda-toolkit/bin/nvcc /opt/cuda/bin/nvcc)
    if(EXISTS ${_nvcc})
      set(CMAKE_CUDA_COMPILER ${_nvcc} CACHE FILEPATH "CUDA compiler")
      break()
    endif()
  endforeach()
endif()

project(ingine LANGUAGES C CXX CUDA)

option(INGINE_BUILD_TESTS "Build ingine tests" OFF)
option(INGINE_BUILD_APPS "Build the ingine executables (serve, bench, validate, hop)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF) # no C++20 modules; skip the dependency scan
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ingine_link=1)
  set(CMAKE_JOB_POOL_LINK ingine_link)
endif()

include(cmake/Dependencies.cmake)

add_subdirectory(third_party)
add_subdirectory(src/core)

# Every other component directory owns its CMakeLists.txt and defines the target named here; a
# directory is added only once it has one, so the tree builds while components are in progress.
# A model pack lives under src/arch/<pack>/ (header root, so its sources still include "model/..." and
# "kernels/...").
#   src/text                        -> ingine_text
#   src/image                       -> ingine_image
#   src/arch/inkling/kernels/<area> -> ingine_kernels_<area>   (aggregated by ingine_kernels)
#   src/arch/inkling/model          -> ingine_model
#   src/arch/qwen35moe              -> qwen35moe_data | qwen35moe_data_check  (header-only, always built)
#   src/arch/qwen38                 -> qwen38_data | qwen38_data_check (always built)
#   src/arch/nemotron_h             -> nemotron_h_data | nemotron_h_data_check (header-only, always built)
#   src/engine                      -> ingine_engine
#   src/serve                       -> ingine_serve
#   apps/serve|bench|validate       -> ingine-serve | ingine-bench | ingine-validate
function(ingine_add_component dir)
  if(EXISTS ${PROJECT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endfunction()

ingine_add_component(src/text)
ingine_add_component(src/image)
# Shared kernels whose shapes more than one pack publishes (currently the Qwen3.x GDN step).
ingine_add_component(src/kernels/gdn)
ingine_add_component(src/kernels/quant)
ingine_add_component(src/kernels/dense)

# One pack is built at a time: Inkling unless -DINGINE_ARCH_PACK names another directory under src/arch. A pack
# that has no sources yet contributes its header-only data target only.
set(INGINE_ARCH_PACK inkling CACHE STRING "Model pack under src/arch (inkling, qwen35moe, qwen38, nemotron_h)")
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/src/arch/${INGINE_ARCH_PACK})
  message(FATAL_ERROR "INGINE_ARCH_PACK=${INGINE_ARCH_PACK}: no such directory under src/arch")
endif()

# Header-only packs (constants and layer tables, no model or kernel sources yet): add_subdirectory always, so their
# headers compile with any pack selected, and nothing from them reaches ingine-serve unless that pack is selected.
set(INGINE_HEADER_ONLY_PACKS qwen35moe qwen38 nemotron_h)
foreach(pack ${INGINE_HEADER_ONLY_PACKS})
  if(EXISTS ${PROJECT_SOURCE_DIR}/src/arch/${pack}/CMakeLists.txt)
    add_subdirectory(src/arch/${pack})
  endif()
endforeach()

set(INGINE_KERNEL_AREAS common dense moe attn head prefill draft vision)
foreach(area ${INGINE_KERNEL_AREAS})
  ingine_add_component(src/arch/${INGINE_ARCH_PACK}/kernels/${area})
endforeach()
add_library(ingine_kernels INTERFACE)
foreach(area ${INGINE_KERNEL_AREAS})
  if(TARGET ingine_kernels_${area})
    target_link_libraries(ingine_kernels INTERFACE ingine_kernels_${area})
  endif()
endforeach()

ingine_add_component(src/arch/${INGINE_ARCH_PACK}/model)
ingine_add_component(src/engine)
ingine_add_component(src/serve)

if(INGINE_BUILD_APPS)
  foreach(app serve bench validate hop)
    ingine_add_component(apps/${app})
  endforeach()
endif()

# A header-only pack reaches the server only through its <pack>_data target, and only when it is the selected pack.
if(TARGET ingine-serve AND TARGET ${INGINE_ARCH_PACK}_data)
  target_link_libraries(ingine-serve PRIVATE ${INGINE_ARCH_PACK}_data)
endif()

if(INGINE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
