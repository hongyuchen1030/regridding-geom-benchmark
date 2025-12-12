#pragma once
#include <vector>
#include <Eigen/Core>

namespace quad {

// Triangle quadrature rule on the *reference triangle* (area = 1/2).
// - w: weights summing to 1/2
// - cs: barycentric coordinates (λ1,λ2,λ3), each row sums to 1
struct TriRule {
  std::vector<double> w;
  std::vector<Eigen::Vector3d> cs;
};

// Returns a symmetric FE2-style rule of requested degree.
// Supported: deg <= 1, 2, 4, 8 (extendible later with more tables).
TriRule get_triangle_fe2_rule(int deg);

} // namespace quad

