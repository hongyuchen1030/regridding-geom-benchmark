#include "eriksson_area.h"

#include <array>
#include <cmath>
#include <vector>

#include "../regrid_benchmark_utils.h"  // operator_helper::vdot/vnorm/cross_fma

namespace {
// Small adapter: Eigen <-> std::array for cross_fma (array-based)
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

// Normalize to unit length (no-op for already-normalized inputs)
inline V3_T<double> vunit(const V3_T<double>& v) {
  const double n = operator_helper::vnorm(v);
  return (n > 0.0) ? (v / n) : v;
}
} // anon

namespace eriksson {

double TriangleAreaFMA(const V3_T<double>& v0,
                       const V3_T<double>& v1,
                       const V3_T<double>& v2) {
  // Ensure unit vectors (robust if inputs are slightly off the unit sphere)
  const V3_T<double> x1 = vunit(v0);
  const V3_T<double> x2 = vunit(v1);
  const V3_T<double> x3 = vunit(v2);

  // Pairwise dots
  const double d12 = operator_helper::vdot(x1, x2);
  const double d23 = operator_helper::vdot(x2, x3);
  const double d31 = operator_helper::vdot(x3, x1);

  // Scalar triple product using FMA-based cross
  const V3_T<double> x2cx3 = vcross_fma(x2, x3);
  const double triple = operator_helper::vdot(x1, x2cx3);

  // Eriksson spherical excess (unsigned area)
  const double num = std::fabs(triple);
  const double den = 1.0 + d12 + d23 + d31;

  // 2*atan2 keeps good conditioning for small/large triangles
  const double E = 2.0 * std::atan2(num, den);

  return E;  // area on the unit sphere (steradians)
}

double SumMeshAreaFMA(const std::vector<Eigen::Vector3d>& xs,
                      const std::vector<Eigen::Vector3i>& elems) {
  double sum = 0.0;
  for (const auto& t : elems) {
    sum += TriangleAreaFMA(xs[t[0]], xs[t[1]], xs[t[2]]);
  }
  return sum;
}

} // namespace eriksson
