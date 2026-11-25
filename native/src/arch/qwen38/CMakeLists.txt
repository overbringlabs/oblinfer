# qwen38: header-only for now. The INTERFACE target carries the pack's include root (so sources keep including
# "model/..." and "kernels/...") and the constants header; the OBJECT target compiles that header on every build.
# Nothing here is linked into ingine-serve unless INGINE_ARCH_PACK=qwen38.
add_library(qwen38_data INTERFACE)
target_include_directories(qwen38_data INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
target_sources(qwen38_data INTERFACE model/model_data.h)

add_library(qwen38_data_check OBJECT model/model_data_check.cpp)
target_link_libraries(qwen38_data_check PRIVATE qwen38_data)
