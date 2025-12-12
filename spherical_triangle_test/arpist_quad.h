// arpist_quad.h
#pragma once
#include <vector>
#include <functional>
#include <Eigen/Core>

namespace arpist {

struct QuadratureResult {
  std::vector<Eigen::Vector3d> points;   // Nqp, on sphere of radius r
  std::vector<double>          weights;  // Nqp, surface-area weights
  std::vector<size_t>          offset;   // nFaces+1, CSR-like
};

// ---------- Baseline (naive triple product) ----------

QuadratureResult compute_sphere_quadrature(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    double h1 = 0.004, int deg1 = 4,
    double h2 = 0.050, int deg2 = 8);

std::vector<double> spherical_integration(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    const std::function<double(const Eigen::Vector3d&)>& f,
    int deg = -1);

// ---------- Accurate (FMA / compensated triple product) ----------

QuadratureResult compute_sphere_quadrature_accu(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    double h1 = 0.004, int deg1 = 4,
    double h2 = 0.050, int deg2 = 8);

std::vector<double> spherical_integration_accu(
    const std::vector<Eigen::Vector3d>& xs,
    const std::vector<Eigen::Vector3i>& elems,
    const std::function<double(const Eigen::Vector3d&)>& f,
    int deg = -1);

} // namespace arpist
