# qwen38: header-only for now. The INTERFACE target carries the pack's include root (so sources keep including
# "model/..." and "kernels/..."), the constants header, and the compile-time shape selector. The OBJECT target
# compiles that header on every build and instantiates all three shapes.
# Nothing here is linked into ingine-serve unless INGINE_ARCH_PACK=qwen38.
set(INGINE_QWEN38_SHAPE 27 CACHE STRING "Qwen3.5/3.8 dense-hybrid shape: 27 (Shape27), 9 (Shape9), or 4 (Shape4)")
set_property(CACHE INGINE_QWEN38_SHAPE PROPERTY STRINGS 27 9 4)
if(NOT INGINE_QWEN38_SHAPE MATCHES "^(27|9|4)$")
  message(FATAL_ERROR "INGINE_QWEN38_SHAPE=${INGINE_QWEN38_SHAPE}: expected 27, 9, or 4")
endif()

add_library(qwen38_data INTERFACE)
target_include_directories(qwen38_data INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
target_sources(qwen38_data INTERFACE model/model_data.h)
target_compile_definitions(qwen38_data INTERFACE INGINE_QWEN38_SHAPE=${INGINE_QWEN38_SHAPE})

add_library(qwen38_data_check OBJECT model/model_data_check.cpp)
target_link_libraries(qwen38_data_check PRIVATE qwen38_data)
