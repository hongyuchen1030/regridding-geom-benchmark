// yac_quad.cpp
#include "yac_quad.h"

#include <array>
#include <cmath>
#include <vector>

#include "../regrid_benchmark_utils.h"  // operator_helper::vdot/vcross/vnorm/cross_fma

namespace {

// Small adapter: Eigen <-> std::array for cross_fma (which is array-based)
inline std::array<double,3> to_array(const V3_T<double>& v) {
  return { v.x(), v.y(), v.z() };
}
inline V3_T<double> from_array(const std::array<double,3>& a) {
  return V3_T<double>(a[0], a[1], a[2]);
}

// FMA cross for Eigen vectors (wraps operator_helper::cross_fma)
inline V3_T<double> vcross_fma(const V3_T<double>& a, const V3_T<double>& b) {
  const auto c = operator_helper::cross_fma<double>(to_array(a), to_array(b));
  return from_array(c);
}

} // anon

namespace yac {

// Direct port of YAC’s dihedral-angle spherical excess (baseline).
double TriangleArea(const V3_T<double>& v0,
                    const V3_T<double>& v1,
                    const V3_T<double>& v2) {
  // Edge plane normals Uij = Vi × Vj
  V3_T<double> u01 = operator_helper::vcross(v0, v1);
  V3_T<double> u12 = operator_helper::vcross(v1, v2);
  V3_T<double> u20 = operator_helper::vcross(v2, v0);

  // Normalize (no degeneracy checks by design)
  const double s01 = operator_helper::vnorm(u01);
  const double s12 = operator_helper::vnorm(u12);
  const double s20 = operator_helper::vnorm(u20);
  u01 /= s01; u12 /= s12; u20 /= s20;

  // Interior angles via dihedral angles between great-circle planes
  const double ca1 = -operator_helper::vdot(u01, u20);
  const double ca2 = -operator_helper::vdot(u12, u01);
  const double ca3 = -operator_helper::vdot(u20, u12);

  const double a1 = std::acos(ca1);
  const double a2 = std::acos(ca2);
  const double a3 = std::acos(ca3);

  // Spherical excess on unit sphere
  constexpr double PI = 3.141592653589793238462643383279502884;
  return (a1 + a2 + a3 - PI);
}

// Compared (FMA) version: only change is using cross_fma for Uij.
double TriangleAreaFMA(const V3_T<double>& v0,
                       const V3_T<double>& v1,
                       const V3_T<double>& v2) {
  V3_T<double> u01 = vcross_fma(v0, v1);
  V3_T<double> u12 = vcross_fma(v1, v2);
  V3_T<double> u20 = vcross_fma(v2, v0);

  const double s01 = operator_helper::vnorm(u01);
  const double s12 = operator_helper::vnorm(u12);
  const double s20 = operator_helper::vnorm(u20);
  u01 /= s01; u12 /= s12; u20 /= s20;

  const double ca1 = -operator_helper::vdot(u01, u20);
  const double ca2 = -operator_helper::vdot(u12, u01);
  const double ca3 = -operator_helper::vdot(u20, u12);

  const double a1 = std::acos(ca1);
  const double a2 = std::acos(ca2);
  const double a3 = std::acos(ca3);

  constexpr double PI = 3.141592653589793238462643383279502884;
  return (a1 + a2 + a3 - PI);
}

double SumMeshArea(const std::vector<Eigen::Vector3d>& xs,
                   const std::vector<Eigen::Vector3i>& elems) {
  double sum = 0.0;
  for (const auto& t : elems) {
    sum += TriangleArea(xs[t[0]], xs[t[1]], xs[t[2]]);
  }
  return sum;
}

double SumMeshAreaFMA(const std::vector<Eigen::Vector3d>& xs,
                      const std::vector<Eigen::Vector3i>& elems) {
  double sum = 0.0;
  for (const auto& t : elems) {
    sum += TriangleAreaFMA(xs[t[0]], xs[t[1]], xs[t[2]]);
  }
  return sum;
}

} // namespace yac
