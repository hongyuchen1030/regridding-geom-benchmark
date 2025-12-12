#pragma once
#include <vector>
#include <Eigen/Core>

// Use your common aliases / helpers
#include "../regrid_benchmark_utils.h"
//   template<typename T> using V3_T  = Eigen::Matrix<T, 3, 1>;
//   template<typename T> using Arc_T = Eigen::Matrix<T, 3, 2>;

namespace tempest {

// -------- Baseline (as before) --------

// Compute spherical triangle area via Tempest’s triangular quadrature,
// using FE2 / Dunavant rules (we only support nOrder=4 or 8 here).
double CalculateTriangleAreaTriQuadrature(
    const V3_T<double>& a,
    const V3_T<double>& b,
    const V3_T<double>& c,
    int nOrder // 4 or 8
);

// Recursive subdivision version (Tempest-style), using
// tol = 0.05 when nOrder>=8; tol = 0.003 otherwise.
double CalculateTriangleAreaTriQuadratureSplit(
    const V3_T<double>& a,
    const V3_T<double>& b,
    const V3_T<double>& c,
    int nOrder // 4 or 8
);

// “Method” wrapper (Tempest GridElements logic for triangles):
// if h < 0.004 -> order 4
// else if h < 0.09 -> order 8
// else split with order 8 and tol=0.05
double CalculateTriangleAreaTriQuadratureMethod(
    const V3_T<double>& a,
    const V3_T<double>& b,
    const V3_T<double>& c
);

// Convenience: total area over a triangle mesh (xs on unit sphere, elems 0-based).
// If deg>0 use fixed degree (4 or 8). If deg<=0 use the Method() per-face.
double SumMeshArea_TriQuadrature(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    int deg // 4, 8, or <=0 for method
);

// -------- Accurate / FMA (new) --------

// Same interfaces as above, but area kernel uses compensated/FMA triple product.
double CalculateTriangleAreaTriQuadratureAccu(
    const V3_T<double>& a,
    const V3_T<double>& b,
    const V3_T<double>& c,
    int nOrder // 4 or 8
);

double CalculateTriangleAreaTriQuadratureSplitAccu(
    const V3_T<double>& a,
    const V3_T<double>& b,
    const V3_T<double>& c,
    int nOrder // 4 or 8
);

double CalculateTriangleAreaTriQuadratureMethodAccu(
    const V3_T<double>& a,
    const V3_T<double>& b,
    const V3_T<double>& c
);

double SumMeshArea_TriQuadratureAccu(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    int deg // 4, 8, or <=0 for method
);

} // namespace tempest
