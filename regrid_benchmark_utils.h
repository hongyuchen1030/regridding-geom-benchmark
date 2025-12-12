#pragma once
#ifndef REGRID_BENCHMARK_UTILS_H
#define REGRID_BENCHMARK_UTILS_H

#include <iostream>
#include <fstream>  
#include <sstream> 
#include <string>
#include <cmath>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <array>
#include <random>
#include <utility>
#include <Eigen/Dense>

#include "simd_fma.hh"

template<typename T>
using V3_T  = Eigen::Matrix<T, 3, 1>;
template<typename T>
using Arc_T = Eigen::Matrix<T, 3, 2>; // 3 rows (coords) × 2 cols (endpoints)

namespace operator_helper {

// Norm / dot for Eigen 3-vectors (double)
inline double vnorm(const Eigen::Vector3d& v) { return v.norm(); }
inline double vdot (const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return a.dot(b); }

// Manual cross (avoid <Eigen/Geometry> dependency)
inline Eigen::Vector3d vcross(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return Eigen::Vector3d(
    a.y()*b.z() - a.z()*b.y(),
    a.z()*b.x() - a.x()*b.z(),
    a.x()*b.y() - a.y()*b.x()
  );
}

// Templated overloads so code that uses V3_T<double> also compiles
template <typename T>
inline T vnorm(const V3_T<T>& v) { return v.norm(); }

template <typename T>
inline T vdot(const V3_T<T>& a, const V3_T<T>& b) { return a.dot(b); }

template <typename T>
inline V3_T<T> vcross(const V3_T<T>& a, const V3_T<T>& b) {
  return V3_T<T>(
    a.y()*b.z() - a.z()*b.y(),
    a.z()*b.x() - a.x()*b.z(),
    a.x()*b.y() - a.y()*b.x()
  );
}

// Max chord-length edge of a triangle (works on unit sphere, cheap & stable)
template <typename Vec3>
inline double max_edge_chord(const Vec3& a, const Vec3& b, const Vec3& c) {
  const double e01 = (a - b).norm();
  const double e12 = (b - c).norm();
  const double e20 = (c - a).norm();
  return std::max(e01, std::max(e12, e20));
}



template <typename T>
inline std::tuple<T, T> two_prod_fma(T a, T b) {
  T x = a * b;
  T y = ::simd_fma(a, b, -x);
  return {x, y};
}

template <typename T>
inline std::tuple<T, T> two_sum(const T& a, const T& b) {
  T x = a + b;
  T z = x - a;
  T y = (a - (x - z)) + (b - z);
  return {x, y};
}

template <typename T>
inline std::tuple<T, T> AccuDOP(T a, T b, T c, T d) {
  auto [p1, s1] = two_prod_fma(a, b);
  auto [h2, r2] = two_prod_fma(c, -d);
  auto [p2, q2] = two_sum(p1, h2);
  T s2 = s1 + (q2 + r2);
  return {p2, s2}; // (hi, lo)
}



template <typename T>
inline T fmms(T a, T b, T c, T d) {
  T cd  = c * d;
  T err = ::simd_fma(-c, d, cd);
  T dop = ::simd_fma(a, b, -cd);
  return dop + err;
}

template <typename T>
inline std::array<T, 3> cross_fma(const std::array<T, 3>& v1,
                                  const std::array<T, 3>& v2) {
  return {
    fmms(v1[1], v2[2], v1[2], v2[1]),
    fmms(v1[2], v2[0], v1[0], v2[2]),
    fmms(v1[0], v2[1], v1[1], v2[0])
  };
}

template <typename T>
inline std::tuple<T, T> two_square_fma(T a) {
  T x = a * a;
  T y = ::simd_fma(a, a, -x);
  return {x, y};
}

template <typename T>
inline std::tuple<T, T> fast_two_sum(T a, T b) {
  T x = a + b;
  T y = (a - x) + b;
  return {x, y};
}

template <typename T>
inline std::tuple<T, T> acc_sqrt_re(T T_val, T t) {
  T P = std::sqrt(T_val);
  auto [H, h] = two_square_fma(P);
  T r = (T_val - H) - h;
  r += t;
  T p = r / (T(2) * P);
  return {P, p};
}

template <typename T, std::size_t N>
inline T dot_fma(const std::array<T, N>& v1, const std::array<T, N>& v2) {
  T s, c;
  std::tie(s, c) = two_prod_fma(v1[0], v2[0]);
  for (std::size_t i = 1; i < N; ++i) {
    T p, pi;
    std::tie(p, pi) = two_prod_fma(v1[i], v2[i]);
    T sigma;
    std::tie(s, sigma) = two_sum(s, p);
    c += pi + sigma;
  }
  return s + c;
}

template <typename T, std::size_t N>
inline std::tuple<T, T> CompDotC(const std::array<T, N>& v1,
                                 const std::array<T, N>& v2) {
  T s, c;
  std::tie(s, c) = two_prod_fma(v1[0], v2[0]);
  for (std::size_t i = 1; i < N; ++i) {
    T p, pi;
    std::tie(p, pi) = two_prod_fma(v1[i], v2[i]);
    T sigma;
    std::tie(s, sigma) = two_sum(s, p);
    c += pi + sigma;
  }
  return {s, c};
}

template <typename T>
inline std::tuple<T, T> sum_non_neg(const std::tuple<T, T>& A,
                                    const std::tuple<T, T>& B) {
  T Ah, Al, Bh, Bl;
  std::tie(Ah, Al) = A;
  std::tie(Bh, Bl) = B;
  auto [H, h] = two_sum(Ah, Bh);
  T c = Al + Bl;
  T d = h + c;
  return fast_two_sum(H, d);
}

template <typename T, std::size_t N>
inline T vec_sum(const std::array<T, N>& arr) {
  T sum = arr[0], compensator = T(0);
  for (std::size_t i = 1; i < N; ++i) {
    auto [partial_sum, err] = two_sum(sum, arr[i]);
    sum = partial_sum;
    compensator += err;
  }
  return sum + compensator;
}

template <typename T, std::size_t N>
inline std::tuple<T, T> SumOfSquaresC(const std::array<T, N>& vals,
                                      const std::array<T, N>& errs) {
  T S = T(0), s = T(0);
  for (std::size_t i = 0; i < N; ++i) {
    auto [P, p] = two_prod_fma(vals[i], vals[i]);
    std::tie(S, s) = sum_non_neg(std::make_tuple(S, s), std::make_tuple(P, p));
  }
  T R = dot_fma(vals, errs);
  T err = T(2) * R + s;
  return fast_two_sum(S, err);
}

// ================================================================
// AccuCross (Algorithm 1): value/error pairs, all in V3_T<T>.
// Returns (vh, vl) as V3_T<T> (hi, lo) of (v1+ev1) × (v2+ev2).
// Each component uses an 8-term compensated dot product (CompDotC).
// ------------------------------------------------
template <typename T>
inline std::pair<V3_T<T>, V3_T<T>>
AccuCross(const V3_T<T>& v1, const V3_T<T>& ev1,
          const V3_T<T>& v2, const V3_T<T>& ev2)
{
  V3_T<T> vh, vl;

  const bool zero_e1 = (ev1[0]==T(0) && ev1[1]==T(0) && ev1[2]==T(0));
  const bool zero_e2 = (ev2[0]==T(0) && ev2[1]==T(0) && ev2[2]==T(0));

  // Fast path: both inputs exact → three AccuDOPs
  if (zero_e1 && zero_e2) {
    T h,l;
    std::tie(h,l) = AccuDOP(v1[1], v2[2], v1[2], v2[1]); vh[0]=h; vl[0]=l; // x = y1*z2 - z1*y2
    std::tie(h,l) = AccuDOP(v1[2], v2[0], v1[0], v2[2]); vh[1]=h; vl[1]=l; // y = z1*x2 - x1*z2
    std::tie(h,l) = AccuDOP(v1[0], v2[1], v1[1], v2[0]); vh[2]=h; vl[2]=l; // z = x1*y2 - y1*x2
    return {vh, vl};
  }

  // General path: expand (v1+e1)×(v2+e2) exactly per component with 8 terms.

  // x = (y1+e1y)(z2+e2z) - (z1+e1z)(y2+e2y)
  {
    // A · B with A from (v1,e1) only, B from (v2,e2) only
    // Terms (in order): y1*z2, y1*e2z, e1y*z2, e1y*e2z,  -z1*y2, -z1*e2y, -e1z*y2, -e1z*e2y
    std::array<T,8> A = {  v1[1],  v1[1],  ev1[1], ev1[1], -v1[2], -v1[2], -ev1[2], -ev1[2] };
    std::array<T,8> B = {  v2[2],  ev2[2], v2[2],  ev2[2],  v2[1],  ev2[1],  v2[1],   ev2[1] };
    T s,c; std::tie(s,c) = CompDotC<T,8>(A, B); vh[0]=s; vl[0]=c;
  }

  // y = (z1+e1z)(x2+e2x) - (x1+e1x)(z2+e2z)
  {
    // Terms: z1*x2, z1*e2x, e1z*x2, e1z*e2x,  -x1*z2, -x1*e2z, -e1x*z2, -e1x*e2z
    std::array<T,8> A = {  v1[2],  v1[2],  ev1[2], ev1[2], -v1[0], -v1[0], -ev1[0], -ev1[0] };
    std::array<T,8> B = {  v2[0],  ev2[0], v2[0],  ev2[0],  v2[2],  ev2[2],  v2[2],   ev2[2] };
    T s,c; std::tie(s,c) = CompDotC<T,8>(A, B); vh[1]=s; vl[1]=c;
  }

  // z = (x1+e1x)(y2+e2y) - (y1+e1y)(x2+e2x)
  {
    // Terms: x1*y2, x1*e2y, e1x*y2, e1x*e2y,  -y1*x2, -y1*e2x, -e1y*x2, -e1y*e2x
    std::array<T,8> A = {  v1[0],  v1[0],  ev1[0], ev1[0], -v1[1], -v1[1], -ev1[1], -ev1[1] };
    std::array<T,8> B = {  v2[1],  ev2[1], v2[1],  ev2[1],  v2[0],  ev2[0],  v2[0],   ev2[0] };
    T s,c; std::tie(s,c) = CompDotC<T,8>(A, B); vh[2]=s; vl[2]=c;
  }

  return {vh, vl};
}

// Convenience overload: zero-error inputs.
template <typename T>
inline std::pair<V3_T<T>, V3_T<T>>
AccuCross(const V3_T<T>& v1, const V3_T<T>& v2)
{
  const V3_T<T> z(T(0), T(0), T(0));
  return AccuCross<T>(v1, z, v2, z);
}



template<typename T>
static V3_T<double> taylor_normalization(std::array<double, 6>& point, std::tuple<double, double> norm) {
    // For each component of the point, multiply by the norm
    V3_T<double> normalized_point;
    double N = std::get<0>(norm);
    double n = std::get<1>(norm);
    for (int i = 0; i < 3; ++i) {
        double S = point[2*i];
        double s = point[2*i+1];
        auto [S1, s1] = two_sum(S, s);
        normalized_point[i] = S1/N+((N*s1 - n * S1)/(N*N));
    }

    return normalized_point;

}

// ===== Small adapters for Eigen / V3_T to avoid rewrites =====

inline std::array<double,3> to_arr(const Eigen::Vector3d& v) {
  return {v.x(), v.y(), v.z()};
}
inline Eigen::Vector3d from_arr(const std::array<double,3>& a) {
  return Eigen::Vector3d(a[0], a[1], a[2]);
}
template<typename T>
inline std::array<T,3> to_arr(const V3_T<T>& v) { return {v.x(), v.y(), v.z()}; }
template<typename T>
inline V3_T<T> from_arr_v3t(const std::array<T,3>& a) { return V3_T<T>(a[0], a[1], a[2]); }

// Cross for Eigen using FMA
inline Eigen::Vector3d vcross_fma_accu(const Eigen::Vector3d& a,
                                       const Eigen::Vector3d& b) {
  return from_arr(cross_fma<double>(to_arr(a), to_arr(b)));
}

// Accurate scalar triple product a · (b × c)
// Build (hi,lo) for each component of b×c via AccuDOP, then do a single
// compensated dot: [a0,a0,a1,a1,a2,a2] · [h0,l0,h1,l1,h2,l2].
inline double triple_product_accu(const Eigen::Vector3d& a,
                                  const Eigen::Vector3d& b,
                                  const Eigen::Vector3d& c)
{
  auto bb = to_arr(b), cc = to_arr(c);

  double h0,l0,h1,l1,h2,l2;
  std::tie(h0,l0) = AccuDOP(bb[1], cc[2], bb[2], cc[1]); // x
  std::tie(h1,l1) = AccuDOP(bb[2], cc[0], bb[0], cc[2]); // y
  std::tie(h2,l2) = AccuDOP(bb[0], cc[1], bb[1], cc[0]); // z

  auto aa = to_arr(a);
  std::array<double,6> avec = { aa[0], aa[0], aa[1], aa[1], aa[2], aa[2] };
  std::array<double,6> hvec = { h0,    l0,    h1,    l1,    h2,    l2    };

  double s,e; std::tie(s,e) = CompDotC<double,6>(avec, hvec);
  return s + e;
}

// Template variant for V3_T<T>
template<typename T>
inline T triple_product_accu(const V3_T<T>& a,
                             const V3_T<T>& b,
                             const V3_T<T>& c)
{
  auto bb = to_arr(b), cc = to_arr(c);

  T h0,l0,h1,l1,h2,l2;
  std::tie(h0,l0) = AccuDOP(bb[1], cc[2], bb[2], cc[1]); // x
  std::tie(h1,l1) = AccuDOP(bb[2], cc[0], bb[0], cc[2]); // y
  std::tie(h2,l2) = AccuDOP(bb[0], cc[1], bb[1], cc[0]); // z

  auto aa = to_arr(a);
  std::array<T,6> avec = { aa[0], aa[0], aa[1], aa[1], aa[2], aa[2] };
  std::array<T,6> hvec = { h0,    l0,    h1,    l1,    h2,    l2    };

  T s,e; std::tie(s,e) = CompDotC<T,6>(avec, hvec);
  return s + e;
}

} // namespace operator_helper



// PrecisionTraits: mantissa bit-width for floating-point types
template<typename T>
struct PrecisionTraits;

template<>
struct PrecisionTraits<float> { static constexpr size_t p = 24; };   // 24 bits for mantissa

template<>
struct PrecisionTraits<double> { static constexpr size_t p = 53; };  // 53 bits for mantissa

// Centralized error logging utility
inline void logError(const std::string& methodName,
                     const std::string& message,
                     double value,
                     int64_t significand,
                     int64_t exponent) {
    std::cerr << "[ERROR] "
              << "Method: " << methodName
              << ", Value: " << value
              << ", Message: " << message
              << ", Significand: " << significand
              << ", Exponent: " << exponent
              << std::endl;
}

// Struct for decomposed floating-point representation
struct DecomposedFloat {
    int64_t significand;
    int64_t exponent;

    DecomposedFloat() : significand(0), exponent(0) {}
    DecomposedFloat(int64_t sig, int64_t exp) : significand(sig), exponent(exp) {}

    // Decompose value v into (significand, exponent) such that
    // v == significand * 2^exponent within machine precision
    template<typename T>
    DecomposedFloat(T v) {
        if (v == T(0)) {
            significand = 0;
            exponent = 0;
        } else {
            // ilogb gives exponent of v in base 2
            exponent = std::ilogb(v) - (std::numeric_limits<T>::digits - 1);
            T normalized = std::ldexp(v, -exponent);  // bring into [1.0, 2.0)
            significand = static_cast<int64_t>(normalized);

            // Validate integer part
            if (static_cast<T>(significand) != normalized) {
                logError("DecomposedFloat", "Non-integer significand!",
                         static_cast<double>(v), significand, exponent);
                throw std::runtime_error("Integer part was not integer!");
            }
            // Validate reconstruction
            if (toFloat<T>() != v) {
                logError("DecomposedFloat", "Reconstruction failure!", 
                         static_cast<double>(v), significand, exponent);
                throw std::runtime_error("Could not reconstruct original value!");
            }
        }
    }

    // Convert back to floating-point
    template<typename T>
    T toFloat() const {
        return std::ldexp(T(significand), exponent);
    }

    // Stream output: "significand * 2^exponent"
    friend std::ostream& operator<<(std::ostream& os, const DecomposedFloat& df) {
        os << df.significand << " * 2^" << df.exponent;
        return os;
    }

    // CSV output: "significand,exponent"
    std::string toCSV() const {
        return std::to_string(significand) + "," + std::to_string(exponent);
    }
};




class RNG {
public:
    explicit RNG(std::uint64_t seed = std::random_device{}())
        : eng_(seed), uni01_(0.0, 1.0) {}

    void reseed(std::uint64_t seed) { eng_.seed(seed); }

    // Uniform [0,1)
    double uniform01() { return uni01_(eng_); }

private:
    std::mt19937_64 eng_;
    std::uniform_real_distribution<double> uni01_;
};

template<typename T>
inline V3_T<T> random_unit_vec(RNG& rng) {
    const T z   = static_cast<T>(2.0 * rng.uniform01() - 1.0);
    const T phi = static_cast<T>(2.0 * M_PI * rng.uniform01());
    const T rxy = std::sqrt(std::max(static_cast<T>(0.0), static_cast<T>(1.0) - z*z));
    return V3_T<T>(rxy * std::cos(phi), rxy * std::sin(phi), z); // already unit
}

template<typename T>
inline V3_T<T> random_vec_at_angle_from(RNG& rng, const V3_T<T>& u_in, T theta_rad) {
    V3_T<T> u = u_in.normalized();
    V3_T<T> r = random_unit_vec<T>(rng);
    if (std::abs(r.dot(u)) > static_cast<T>(0.9999))
        r = random_unit_vec<T>(rng);
    V3_T<T> w = (r - (r.dot(u)) * u).normalized();
    const T c = std::cos(theta_rad);
    const T s = std::sin(theta_rad);
    return (c * u + s * w).normalized();
}

template<typename T>
inline T angle_deg_between_unit(const V3_T<T>& u, const V3_T<T>& v) {
    T c = u.dot(v);
    if (c > static_cast<T>(1.0))  c = static_cast<T>(1.0);
    if (c < static_cast<T>(-1.0)) c = static_cast<T>(-1.0);
    return std::acos(c) * static_cast<T>(180.0 / M_PI);
}

template<typename T>
struct VecPair {
    V3_T<T> u;
    V3_T<T> v;
    T angle_deg;
};

template<typename T>
inline std::vector<VecPair<T>>
generate_pairs_0_90_with_ill_coverage(std::uint64_t seed,
                                      std::size_t total_N,
                                      std::size_t ill_count)
{
    if (total_N < 2 * ill_count) {
        throw std::invalid_argument("total_N must be at least 2 * ill_count");
    }

    RNG rng(seed);
    std::vector<VecPair<T>> out;
    out.reserve(total_N + 2 * ill_count + 64);

    auto make_pair_at_deg = [&](T theta_deg) -> VecPair<T> {
        auto u = random_unit_vec<T>(rng);
        auto v = random_vec_at_angle_from<T>(rng, u, theta_deg * M_PI / static_cast<T>(180.0));
        return {u, v, theta_deg};
    };

    // Bulk sampling over [0,90]
    for (std::size_t i = 0; i < total_N; ++i) {
        T theta_deg = static_cast<T>(90.0) * rng.uniform01();
        out.push_back(make_pair_at_deg(theta_deg));
    }

    // Ensure ill-conditioned bands have enough samples
    const T LO1 = 0.0, HI1 = 1.0;    // near 0°
    const T LO2 = 89.0, HI2 = 90.0;  // near 90°

    auto count_in_band = [&](T lo, T hi) {
        return static_cast<std::size_t>(std::count_if(
            out.begin(), out.end(),
            [&](const VecPair<T>& p) { return p.angle_deg >= lo && p.angle_deg <= hi; }
        ));
    };

    std::size_t c_band1 = count_in_band(LO1, HI1);
    std::size_t c_band2 = count_in_band(LO2, HI2);

    while (c_band1 < ill_count) {
        T theta_deg = LO1 + (HI1 - LO1) * rng.uniform01();
        out.push_back(make_pair_at_deg(theta_deg));
        ++c_band1;
    }
    while (c_band2 < ill_count) {
        T theta_deg = LO2 + (HI2 - LO2) * rng.uniform01();
        out.push_back(make_pair_at_deg(theta_deg));
        ++c_band2;
    }

    //Trim overshoot ONLY from the 30–60° band
    if (out.size() > total_N) {
        std::size_t need = out.size() - total_N;

        // collect indices of removable elements (in mid-band only)
        std::vector<std::size_t> mid_idx;
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (out[i].angle_deg >= 30.0 && out[i].angle_deg <= 60.0) {
                mid_idx.push_back(i);
            }
        }

        // remove from the back of mid_idx to avoid shifting many elements early
        if (need > mid_idx.size()) need = mid_idx.size();
        for (std::size_t k = 0; k < need; ++k) {
            std::size_t idx = mid_idx[mid_idx.size() - 1 - k];
            out.erase(out.begin() + static_cast<std::ptrdiff_t>(idx));
        }
    }

    // Final sort and filter just in case
    std::sort(out.begin(), out.end(),
              [](const VecPair<T>& a, const VecPair<T>& b) { return a.angle_deg < b.angle_deg; });
    out.erase(std::remove_if(out.begin(), out.end(),
                             [](const VecPair<T>& p) {
                                 return p.angle_deg < 0.0 || p.angle_deg > 90.0;
                             }),
              out.end());

    return out;
}





// Load arcs from CSV written by GCR_generate_arcs.cpp
// Columns:
//  0: ref_degree
//  1,2: x1_significand(int64), x1_exponent(int64)
//  3,4: y1_significand(int64), y1_exponent(int64)
//  5,6: z1_significand(int64), z1_exponent(int64)
//  7,8: x2_significand(int64), x2_exponent(int64)
//  9,10: y2_significand(int64), y2_exponent(int64)
//  11,12: z2_significand(int64), z2_exponent(int64)
template<typename T>
static bool load_arcs_csv(const char* path, std::vector<VecPair<T>>& out) {
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "Failed to open input CSV: " << path << "\n";
        return false;
    }

    std::string line;
    // Skip header
    if (!std::getline(fin, line)) {
        std::cerr << "Empty file or failed to read header: " << path << "\n";
        return false;
    }

    out.clear();

    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string tok;

        auto next = [&](std::string& dst)->bool {
            if (!std::getline(ss, dst, ',')) return false;
            return true;
        };

        VecPair<T> row{};
        std::string s;

        // ref_degree -> angle_deg
        if (!next(s)) break;
        row.angle_deg = static_cast<T>(std::strtod(s.c_str(), nullptr));

        // Helper to read (sig,exp) pair as int64 and convert via DecomposedFloat
        auto read_sigexp_to = [&](T& dst)->bool {
            std::string sigs, exps;
            if (!next(sigs) || !next(exps)) return false;
            // parse as int64_t (as defined in your DecomposedFloat)
            errno = 0;
            long long sig_ll = std::strtoll(sigs.c_str(), nullptr, 10);
            long long exp_ll = std::strtoll(exps.c_str(), nullptr, 10);
            if (errno != 0) return false;
            DecomposedFloat df(static_cast<int64_t>(sig_ll),
                               static_cast<int64_t>(exp_ll));
            dst = df.toFloat<T>();
            return true;
        };

        // x1, y1, z1
        if (!read_sigexp_to(row.u[0])) break;
        if (!read_sigexp_to(row.u[1])) break;
        if (!read_sigexp_to(row.u[2])) break;

        // x2, y2, z2
        if (!read_sigexp_to(row.v[0])) break;
        if (!read_sigexp_to(row.v[1])) break;
        if (!read_sigexp_to(row.v[2])) break;

        out.push_back(std::move(row));
    }

    if (out.empty()) {
        std::cerr << "No data rows loaded from: " << path << "\n";
        return false;
    }
    return true;
}



#endif // REGRID_BENCHMARK_UTILS_H
