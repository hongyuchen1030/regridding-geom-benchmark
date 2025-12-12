// arpist_quad.cpp
#include "arpist_quad.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stack>
#include <vector>

#include <Eigen/Core>

#include "triangle_quadrature_rules.h"
#include "../regrid_benchmark_utils.h"   // operator_helper::vdot, vcross, max_edge_chord, triple_product_accu

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace {

// Quadrature rule (barycentric points on reference triangle)
using Rule = quad::TriRule;
inline Rule get_rule(int deg) { return quad::get_triangle_fe2_rule(deg); }

// Simple growable buffers
struct GrowBuffers {
  std::vector<Eigen::Vector3d>& P;
  std::vector<double>&          W;
  size_t index{0};

  void ensure(size_t need) {
    if (index + need > W.size()) {
      const size_t newN = std::max(W.size()*2 + need, need);
      W.resize(newN);
      P.resize(newN);
    }
  }
  void push(const Eigen::Vector3d& p, double w) {
    ensure(1);
    P[index] = p;
    W[index] = w;
    ++index;
  }
};

// ----- One triangle, baseline (naive triple product; ANCHORED) -----
template <class Grow>
inline void quad_sphere_tri_naive(const Eigen::Vector3d tri[3], int deg, Grow& grow)
{
  // Anchor at x0: dot(x0, (x1 - x0) × (x2 - x0))  
  const Eigen::Vector3d e1 = tri[1] - tri[0];
  const Eigen::Vector3d e2 = tri[2] - tri[0];
  const double tri_pro = std::abs(
      operator_helper::vdot(tri[0], operator_helper::vcross(e1, e2)));

  const Rule rule = get_rule(deg);
  for (size_t q = 0; q < rule.w.size(); ++q) {
    const Eigen::Vector3d b = rule.cs[q];  // (λ1,λ2,λ3), sums to 1
    Eigen::Vector3d p = b[0]*tri[0] + b[1]*tri[1] + b[2]*tri[2];
    const double nrm = p.norm();
    p /= nrm;  // project to unit sphere
    const double w = rule.w[q] * (tri_pro / (nrm*nrm*nrm));  // Jacobian
    grow.push(p, w);
  }
}


// ----- One triangle, accurate (compensated/FMA triple product; ANCHORED) -----
template <class Grow>
inline void quad_sphere_tri_accu(const Eigen::Vector3d tri[3], int deg, Grow& grow)
{
  // Same anchor as above, but use our compensated/FMA triple product:
  // triple_product_accu(a, b, c) ≈ a · (b × c) with EFT compensation.
  const Eigen::Vector3d e1 = tri[1] - tri[0];
  const Eigen::Vector3d e2 = tri[2] - tri[0];
  const double tri_pro = std::abs(
      operator_helper::triple_product_accu(tri[0], e1, e2));

  const Rule rule = get_rule(deg);
  for (size_t q = 0; q < rule.w.size(); ++q) {
    const Eigen::Vector3d b = rule.cs[q];
    Eigen::Vector3d p = b[0]*tri[0] + b[1]*tri[1] + b[2]*tri[2];
    const double nrm = p.norm();
    p /= nrm;
    const double w = rule.w[q] * (tri_pro / (nrm*nrm*nrm));
    grow.push(p, w);
  }
}


// ----- Split until edges <= tol, then call the given kernel -----
template <class Grow, void(*Kernel)(const Eigen::Vector3d[3], int, Grow&)>
void quad_sphere_tri_split(const Eigen::Vector3d tri_in[3], double tol, int deg, Grow& grow)
{
  struct T { Eigen::Vector3d a,b,c; };
  std::stack<T> st;
  st.push({tri_in[0], tri_in[1], tri_in[2]});
  while (!st.empty()) {
    const T cur = st.top(); st.pop();
    Eigen::Vector3d tri[3] = {cur.a, cur.b, cur.c};

    if (operator_helper::max_edge_chord(tri[0], tri[1], tri[2]) > tol) {
      // midpoints on sphere
      const Eigen::Vector3d m01 = (tri[0]+tri[1]).normalized();
      const Eigen::Vector3d m12 = (tri[1]+tri[2]).normalized();
      const Eigen::Vector3d m20 = (tri[2]+tri[0]).normalized();

      // same 4-way split as before
      st.push({tri[0], m01,   m20});
      st.push({m20,    m01,   m12});
      st.push({m12,    m01,   tri[1]});
      st.push({m20,    m12,   tri[2]});
    } else {
      Kernel(tri, deg, grow);
    }
  }
}

} // anonymous

namespace arpist {

// ===============================
// Baseline (naive)
// ===============================
QuadratureResult compute_sphere_quadrature(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    double h1, int deg1, double h2, int deg2)
{
  if (xs.empty()) throw std::runtime_error("compute_sphere_quadrature: empty vertex array");

  QuadratureResult out;
  out.offset.resize(elems.size() + 1, 0);

  // radius r from first vertex, check all on same sphere
  const double r = operator_helper::vnorm(xs.front());
  for (const auto& v : xs) {
    if (std::abs(operator_helper::vnorm(v) - r) > 2e-6)
      throw std::runtime_error("Input vertices not on a common sphere.");
  }

  const size_t nf = elems.size();
  out.points.resize(std::max<size_t>(1'000'000, nf*6));
  out.weights.resize(out.points.size());
  GrowBuffers gb{out.points, out.weights};

  for (size_t f = 0; f < nf; ++f) {
    out.offset[f] = gb.index;

    const Eigen::Vector3i t = elems[f];
    Eigen::Vector3d tri[3] = { xs[t[0]]/r, xs[t[1]]/r, xs[t[2]]/r };

    const double h = operator_helper::max_edge_chord(tri[0], tri[1], tri[2]);
    if (h < h1) {
      quad_sphere_tri_naive(tri, deg1, gb);
    } else if (h < h2) {
      quad_sphere_tri_naive(tri, deg2, gb);
    } else {
      quad_sphere_tri_split<GrowBuffers, quad_sphere_tri_naive>(tri, h2, deg2, gb);
    }
  }

  out.offset[nf] = gb.index;

  // scale to radius r (weights scale with r^2)
  out.points.resize(gb.index);
  out.weights.resize(gb.index);
  for (size_t i = 0; i < gb.index; ++i) out.points[i] *= r;
  const double scale = r*r;
  for (size_t i = 0; i < gb.index; ++i) out.weights[i] *= scale;

  return out;
}

std::vector<double> spherical_integration(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    const std::function<double(const Eigen::Vector3d&)>& f,
    int deg)
{
  QuadratureResult Q = (deg>0)
    ? compute_sphere_quadrature(xs, elems, /*h1*/100, deg,100,deg)// disable branching now
    : compute_sphere_quadrature(xs, elems); // default h1/deg1/h2/deg2

  std::vector<double> face_vals(elems.size(), 0.0);

  #pragma omp parallel for if(elems.size() > 1024)
  for (ptrdiff_t fid = 0; fid < static_cast<ptrdiff_t>(elems.size()); ++fid) {
    double acc = 0.0;
    const size_t a = Q.offset[fid], b = Q.offset[fid+1];
    for (size_t i = a; i < b; ++i) acc += f(Q.points[i]) * Q.weights[i];
    face_vals[static_cast<size_t>(fid)] = acc;
  }
  return face_vals;
}

// ===============================
// Accurate (FMA / compensated)
// ===============================
QuadratureResult compute_sphere_quadrature_accu(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    double h1, int deg1, double h2 , int deg2)
{
  if (xs.empty()) throw std::runtime_error("compute_sphere_quadrature_accu: empty vertex array");

  QuadratureResult out;
  out.offset.resize(elems.size() + 1, 0);

  // radius r from first vertex, check all on same sphere
  const double r = operator_helper::vnorm(xs.front());
  for (const auto& v : xs) {
    if (std::abs(operator_helper::vnorm(v) - r) > 2e-6)
      throw std::runtime_error("Input vertices not on a common sphere.");
  }

  const size_t nf = elems.size();
  out.points.resize(std::max<size_t>(1'000'000, nf*6));
  out.weights.resize(out.points.size());
  GrowBuffers gb{out.points, out.weights};

  for (size_t f = 0; f < nf; ++f) {
    out.offset[f] = gb.index;

    const Eigen::Vector3i t = elems[f];
    Eigen::Vector3d tri[3] = { xs[t[0]]/r, xs[t[1]]/r, xs[t[2]]/r };

    const double h = operator_helper::max_edge_chord(tri[0], tri[1], tri[2]);
    if (h < h1) {
      quad_sphere_tri_accu(tri, deg1, gb);
    } else if (h < h2) {
      quad_sphere_tri_accu(tri, deg2, gb);
    } else {
      quad_sphere_tri_split<GrowBuffers, quad_sphere_tri_accu>(tri, h2, deg2, gb);
    }
  }

  out.offset[nf] = gb.index;

  // scale to radius r (weights scale with r^2)
  out.points.resize(gb.index);
  out.weights.resize(gb.index);
  for (size_t i = 0; i < gb.index; ++i) out.points[i] *= r;
  const double scale = r*r;
  for (size_t i = 0; i < gb.index; ++i) out.weights[i] *= scale;

  return out;
}

std::vector<double> spherical_integration_accu(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    const std::function<double(const Eigen::Vector3d&)>& f,
    int deg)
{
  QuadratureResult Q = (deg>0)
    ? compute_sphere_quadrature_accu(xs, elems, /*h1*/100.0, deg, /*h2*/100.0, deg)
    : compute_sphere_quadrature_accu(xs, elems); // default h1/deg1/h2/deg2

  std::vector<double> face_vals(elems.size(), 0.0);

  #pragma omp parallel for if(elems.size() > 1024)
  for (ptrdiff_t fid = 0; fid < static_cast<ptrdiff_t>(elems.size()); ++fid) {
    double acc = 0.0;
    const size_t a = Q.offset[fid], b = Q.offset[fid+1];
    for (size_t i = a; i < b; ++i) acc += f(Q.points[i]) * Q.weights[i];
    face_vals[static_cast<size_t>(fid)] = acc;
  }
  return face_vals;
}

} // namespace arpist
