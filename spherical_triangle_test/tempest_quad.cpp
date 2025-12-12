// tempest_quad.cpp
#include "tempest_quad.h"
#include "triangle_quadrature_rules.h"

#include <stack>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <vector>

#include "../regrid_benchmark_utils.h"  // operator_helper::vnorm/vdot/vcross/max_edge_chord/triple_product_accu

namespace {

// Integrate one spherical triangle with FE2 rule (weights sum to 1/2 in our tables)
// Baseline (naive triple product)
inline double area_tri_fe2_naive(const V3_T<double>& a,
                                 const V3_T<double>& b,
                                 const V3_T<double>& c,
                                 int nOrder)
{
  if (!(nOrder == 4 || nOrder == 8)) {
    throw std::runtime_error("Tempest FE2 area (naive): nOrder must be 4 or 8");
  }

  const auto rule = quad::get_triangle_fe2_rule(nOrder);

  // |a · (b × c)|
  const double tri_pro =
      std::abs(operator_helper::vdot(a, operator_helper::vcross(b, c)));

  double accum = 0.0;
  for (size_t q = 0; q < rule.w.size(); ++q) {
    const auto& lam = rule.cs[q]; // (λ1,λ2,λ3) sums to 1
    V3_T<double> p = lam[0]*a + lam[1]*b + lam[2]*c;
    const double nrm = operator_helper::vnorm(p);
    // FE2 weights already sum to 1/2 on the reference triangle.
    accum += rule.w[q] * (tri_pro / (nrm*nrm*nrm));
  }
  return accum;
}

// Integrate one spherical triangle with FE2 rule (accurate / compensated triple product)
inline double area_tri_fe2_accu(const V3_T<double>& a,
                                const V3_T<double>& b,
                                const V3_T<double>& c,
                                int nOrder)
{
  if (!(nOrder == 4 || nOrder == 8)) {
    throw std::runtime_error("Tempest FE2 area (accu): nOrder must be 4 or 8");
  }

  const auto rule = quad::get_triangle_fe2_rule(nOrder);

  // |a · (b × c)| with compensated/FMA path
  const double tri_pro = std::abs(operator_helper::triple_product_accu(a, b, c));

  double accum = 0.0;
  for (size_t q = 0; q < rule.w.size(); ++q) {
    const auto& lam = rule.cs[q];
    V3_T<double> p = lam[0]*a + lam[1]*b + lam[2]*c;
    const double nrm = operator_helper::vnorm(p);
    accum += rule.w[q] * (tri_pro / (nrm*nrm*nrm));
  }
  return accum;
}

} // anonymous namespace

namespace tempest {

// ---------------- Baseline (as before) ----------------

double CalculateTriangleAreaTriQuadrature(
    const V3_T<double>& a,
    const V3_T<double>& b,
    const V3_T<double>& c,
    int nOrder
) {
  return area_tri_fe2_naive(a,b,c,nOrder);
}

double CalculateTriangleAreaTriQuadratureSplit(
    const V3_T<double>& a0,
    const V3_T<double>& b0,
    const V3_T<double>& c0,
    int nOrder
) {
  const double tol = (nOrder >= 8) ? 0.05 : 0.004;

  struct Tri { V3_T<double> a,b,c; };
  std::stack<Tri> st;
  st.push({a0,b0,c0});

  double area = 0.0;
  while (!st.empty()) {
    auto [a,b,c] = st.top(); st.pop();
    if (operator_helper::max_edge_chord(a,b,c) > tol) {
      const V3_T<double> m01 = (a+b).normalized();
      const V3_T<double> m12 = (b+c).normalized();
      const V3_T<double> m20 = (c+a).normalized();
      st.push({a,   m01, m20});
      st.push({m20, m01, m12});
      st.push({m12, m01, b  });
      st.push({m20, m12, c  });
    } else {
      area += area_tri_fe2_naive(a,b,c,nOrder);
    }
  }
  return area;
}

double CalculateTriangleAreaTriQuadratureMethod(
    const V3_T<double>& a,
    const V3_T<double>& b,
    const V3_T<double>& c
) {
  const double h = operator_helper::max_edge_chord(a,b,c);
  if (h < 0.004) {
    return area_tri_fe2_naive(a,b,c,4);
  } else if (h < 0.09) {
    return area_tri_fe2_naive(a,b,c,8);
  } else {
    return CalculateTriangleAreaTriQuadratureSplit(a,b,c,8);
  }
}

double SumMeshArea_TriQuadrature(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    int deg
) {
  double sum = 0.0;
  for (const auto& t : elems) {
    const V3_T<double> a = xs[t[0]];
    const V3_T<double> b = xs[t[1]];
    const V3_T<double> c = xs[t[2]];
    if (deg > 0) {
      sum += CalculateTriangleAreaTriQuadrature(a,b,c,deg);
    } else {
      sum += CalculateTriangleAreaTriQuadratureMethod(a,b,c);
    }
  }
  return sum;
}

// ---------------- Accurate / FMA (new) ----------------

double CalculateTriangleAreaTriQuadratureAccu(
    const V3_T<double>& a,
    const V3_T<double>& b,
    const V3_T<double>& c,
    int nOrder
) {
  return area_tri_fe2_accu(a,b,c,nOrder);
}

double CalculateTriangleAreaTriQuadratureSplitAccu(
    const V3_T<double>& a0,
    const V3_T<double>& b0,
    const V3_T<double>& c0,
    int nOrder
) {
  const double tol = (nOrder >= 8) ? 0.05 : 0.003;

  struct Tri { V3_T<double> a,b,c; };
  std::stack<Tri> st;
  st.push({a0,b0,c0});

  double area = 0.0;
  while (!st.empty()) {
    auto [a,b,c] = st.top(); st.pop();
    if (operator_helper::max_edge_chord(a,b,c) > tol) {
      const V3_T<double> m01 = (a+b).normalized();
      const V3_T<double> m12 = (b+c).normalized();
      const V3_T<double> m20 = (c+a).normalized();
      st.push({a,   m01, m20});
      st.push({m20, m01, m12});
      st.push({m12, m01, b  });
      st.push({m20, m12, c  });
    } else {
      area += area_tri_fe2_accu(a,b,c,nOrder);
    }
  }
  return area;
}

double CalculateTriangleAreaTriQuadratureMethodAccu(
    const V3_T<double>& a,
    const V3_T<double>& b,
    const V3_T<double>& c
) {
  const double h = operator_helper::max_edge_chord(a,b,c);
  if (h < 0.004) {
    return area_tri_fe2_accu(a,b,c,4);
  } else if (h < 0.09) {
    return area_tri_fe2_accu(a,b,c,8);
  } else {
    return CalculateTriangleAreaTriQuadratureSplitAccu(a,b,c,8);
  }
}

double SumMeshArea_TriQuadratureAccu(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    int deg
) {
  double sum = 0.0;
  for (const auto& t : elems) {
    const V3_T<double> a = xs[t[0]];
    const V3_T<double> b = xs[t[1]];
    const V3_T<double> c = xs[t[2]];
    if (deg > 0) {
      sum += CalculateTriangleAreaTriQuadratureAccu(a,b,c,deg);
    } else {
      sum += CalculateTriangleAreaTriQuadratureMethodAccu(a,b,c);
    }
  }
  return sum;
}

} // namespace tempest
