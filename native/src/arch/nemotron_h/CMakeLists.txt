# nemotron_h: header-only for now. The INTERFACE target carries the pack's include root (so sources keep including
# "model/..." and "kernels/...") and the constants header; the OBJECT target compiles that header on every build.
# Nothing here is linked into ingine-serve unless INGINE_ARCH_PACK=nemotron_h.
add_library(nemotron_h_data INTERFACE)
target_include_directories(nemotron_h_data INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
target_sources(nemotron_h_data INTERFACE model/model_data.h)

add_library(nemotron_h_data_check OBJECT model/model_data_check.cpp)
target_link_libraries(nemotron_h_data_check PRIVATE nemotron_h_data)
