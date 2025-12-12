#pragma once
#include <vector>
#include <Eigen/Core>
#include "../regrid_benchmark_utils.h"  // V3_T / Arc_T and operator_helper helpers

namespace yac {

// Spherical triangle area on the unit sphere using YAC's dihedral-angles
// formula (ICON-origin). No degeneracy checks here (matches the original).
double TriangleArea(const V3_T<double>& v0,
                    const V3_T<double>& v1,
                    const V3_T<double>& v2);

// “Compared” variant: same formula, but edge-plane normals use the
// FMA-based cross product (cross_fma). Everything else unchanged.
double TriangleAreaFMA(const V3_T<double>& v0,
                       const V3_T<double>& v1,
                       const V3_T<double>& v2);

// Sum of triangle areas over a mesh (xs assumed on unit sphere; elems 0-based).
double SumMeshArea(const std::vector<Eigen::Vector3d>& xs,
                   const std::vector<Eigen::Vector3i>& elems);

// Same as above, but per-face area uses TriangleAreaFMA.
double SumMeshAreaFMA(const std::vector<Eigen::Vector3d>& xs,
                      const std::vector<Eigen::Vector3i>& elems);

} // namespace yac
