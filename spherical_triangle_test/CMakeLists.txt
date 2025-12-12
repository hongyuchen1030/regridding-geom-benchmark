cmake_minimum_required(VERSION 3.18)
project(spherical_triangle_test CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BENCH_FAST_MATH "Enable -ffast-math for perf (unsafe for accuracy)" OFF)
option(ENABLE_OPENMP    "Enable OpenMP" OFF)

# Where shared headers live (e.g., regrid_benchmark_utils.h, simd_fma.hh)
set(SHARED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# ------------ Per-target FP flags helper (define BEFORE use) ------------
function(apply_fp_flags tgt)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    if(BENCH_FAST_MATH)
      target_compile_options(${tgt} PRIVATE -O3 -march=native -ffast-math)
    else()
      target_compile_options(${tgt} PRIVATE -O3 -march=native -ffp-contract=off -fno-fast-math)
    endif()
  endif()
endfunction()

# ------------ Eigen ------------
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen include: ${EIGEN3_INCLUDE_DIR}")

# ------------ OpenMP (optional) ------------
if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# =========================================================
# Spherical triangle quadrature module (general-purpose)
# =========================================================
set(SPHTRI_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/arpist_quad.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/triangle_quadrature_rules.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tempest_quad.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/yac_quad.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eriksson_area.cpp
)
set(SPHTRI_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/arpist_quad.h
  ${CMAKE_CURRENT_SOURCE_DIR}/triangle_quadrature_rules.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tempest_quad.h
  ${CMAKE_CURRENT_SOURCE_DIR}/yac_quad.h
  ${CMAKE_CURRENT_SOURCE_DIR}/eriksson_area.h
)

add_library(sphtri STATIC ${SPHTRI_SOURCES} ${SPHTRI_HEADERS})
target_include_directories(sphtri PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}   # local headers
  ${SHARED_DIR}                 # for regrid_benchmark_utils.h
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(sphtri PUBLIC Eigen3::Eigen)
if(ENABLE_OPENMP)
  target_link_libraries(sphtri PUBLIC OpenMP::OpenMP_CXX)
endif()
apply_fp_flags(sphtri)

# =========================================================
# Executables
# =========================================================

# test_oversphere: main benchmark driver using sphtri
add_executable(test_oversphere
  ${CMAKE_CURRENT_SOURCE_DIR}/test_oversphere.cpp
)
target_include_directories(test_oversphere PRIVATE
  ${SHARED_DIR}
)
target_link_libraries(test_oversphere PRIVATE
  sphtri
)
apply_fp_flags(test_oversphere)

# Generate ill-conditioned spherical triangles -> CSV
add_executable(generate_ill_triangles
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_ill_triangles.cpp
)
target_include_directories(generate_ill_triangles PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}   # local headers
  ${SHARED_DIR}                 # for regrid_benchmark_utils.h, simd_fma.hh
  ${EIGEN3_INCLUDE_DIR}         # Eigen
)
target_link_libraries(generate_ill_triangles PRIVATE Eigen3::Eigen)
apply_fp_flags(generate_ill_triangles)

# decompose_mesh_vertices -> CSV (19-column sig/exp)
add_executable(decompose_mesh_vertices
  ${CMAKE_CURRENT_SOURCE_DIR}/decompose_mesh_vertices.cpp
)
target_include_directories(decompose_mesh_vertices PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}   # local includes if needed
  ${SHARED_DIR}                 # regrid_benchmark_utils.h, simd_fma.hh
  ${EIGEN3_INCLUDE_DIR}         # (harmless even if not used)
)
target_link_libraries(decompose_mesh_vertices PRIVATE Eigen3::Eigen)
apply_fp_flags(decompose_mesh_vertices)
