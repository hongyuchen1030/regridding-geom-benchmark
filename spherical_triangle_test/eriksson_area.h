#pragma once
#include <vector>
#include <Eigen/Core>
#include "../regrid_benchmark_utils.h"  // V3_T / Arc_T and operator_helper helpers

namespace eriksson {

double TriangleAreaFMA(const V3_T<double>& v0,
                       const V3_T<double>& v1,
                       const V3_T<double>& v2);

double SumMeshAreaFMA(const std::vector<Eigen::Vector3d>& xs,
                      const std::vector<Eigen::Vector3i>& elems);

} // namespace eriksson
