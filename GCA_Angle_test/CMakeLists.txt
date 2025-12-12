cmake_minimum_required(VERSION 3.10)
project(regrid_benchmark_formula_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Paths
set(TEST_DIR   ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)  # contains regrid_benchmark_utils.h

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# === Toggle: accuracy-safe vs max-performance floating point ===
# OFF (default) -> accuracy-safe: -fno-fast-math, -ffp-contract=off
# ON             -> max-performance: -ffast-math (enables reassociation etc.)
option(BENCH_FAST_MATH "Enable fast-math for performance benchmarks" OFF)

# ---- Eigen (header-only) ----
if(NOT DEFINED EIGEN3_INCLUDE_DIR AND DEFINED ENV{EIGEN3_INCLUDE_DIR})
  set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" CACHE PATH "Eigen3 include dir")
endif()
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
  set(EIGEN3_INCLUDE_DIR "/global/homes/h/hyvchen/Regrid_Benchmark/extern/eigen-3.4.0"
      CACHE PATH "Eigen3 include dir")
endif()

add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
)

find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)


# Shared headers (so #include "regrid_benchmark_utils.h" works)
include_directories(${SHARED_DIR})

# ---- Tests ----
set(TEST_SOURCES
  generate_arcs.cpp
  GCR_distance_test.cpp
  GCR_angle_perf.cpp
)

foreach(src_file IN LISTS TEST_SOURCES)
  get_filename_component(target ${src_file} NAME_WE)
  add_executable(${target} ${src_file})
  target_include_directories(${target} PRIVATE ${SHARED_DIR})
  target_link_libraries(${target} PRIVATE Eigen3::Eigen)

  # Per-target FP options (GCC/Clang). Keeps other compilers untouched.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    if(BENCH_FAST_MATH)
      # Max performance (not accuracy-safe)
      target_compile_options(${target} PRIVATE
        -O3
        -march=native
        -ffast-math
      )
    else()
      # Accuracy-safe defaults
      target_compile_options(${target} PRIVATE
        -O3
        -march=native
        -ffp-contract=off
        -fno-fast-math
      )
    endif()
  endif()
endforeach()

target_link_libraries(GCR_angle_perf PRIVATE OpenMP::OpenMP_CXX TBB::tbb)
