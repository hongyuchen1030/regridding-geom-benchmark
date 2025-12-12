cmake_minimum_required(VERSION 3.18)
project(GCAGCA_intersection CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Toggle if you ever want to benchmark with -ffast-math (default OFF for accuracy)
option(BENCH_FAST_MATH "Enable -ffast-math for perf (unsafe for accuracy)" OFF)

# Where shared headers live (regrid_benchmark_utils.h, simd_fma.hh)
set(SHARED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# ---------------- FP flags (same style as reference) ----------------
function(apply_fp_flags tgt)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    if(BENCH_FAST_MATH)
      target_compile_options(${tgt} PRIVATE -O3 -march=native -ffast-math)
    else()
      # disable fast-math and disable FMA contraction (keep IEEE semantics)
      target_compile_options(${tgt} PRIVATE -O3 -march=native -ffp-contract=off -fno-fast-math)
    endif()
  endif()
endfunction()

# ---------------- Eigen ----------------
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen include: ${EIGEN3_INCLUDE_DIR}")

# ===================== Executables =====================

add_executable(generate_gca_gca
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_gca_gca.cpp
)
target_include_directories(generate_gca_gca PRIVATE
  ${SHARED_DIR}            # for regrid_benchmark_utils.h, simd_fma.hh
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(generate_gca_gca PRIVATE Eigen3::Eigen)
apply_fp_flags(generate_gca_gca)

add_executable(test_GCA_GCA
  ${CMAKE_CURRENT_SOURCE_DIR}/test_GCA_GCA.cpp
)
target_include_directories(test_GCA_GCA PRIVATE
  ${SHARED_DIR}            # for regrid_benchmark_utils.h, simd_fma.hh
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test_GCA_GCA PRIVATE Eigen3::Eigen)
apply_fp_flags(test_GCA_GCA)
