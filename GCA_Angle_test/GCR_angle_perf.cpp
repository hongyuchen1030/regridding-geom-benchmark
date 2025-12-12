// Build (CMake target already links Eigen and uses -O3 -fno-fast-math -ffp-contract=off):
//   cmake --build . --config Release
//
// Run on Perlmutter CPU node (example; SERIAL):
//   srun -C cpu -N 1 -n 1 -c 1 ./GCR_angle_perf_serial
//
// Output CSV:
//   /global/homes/h/hyvchen/Regrid_Benchmark/src/GCR_Angle_test/results/angle_throughput_serial.csv

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <cassert>
#include <Eigen/Dense>
#include "regrid_benchmark_utils.h"  // V3_T<T>, Arc_T<T>, Eigen::MatrixXd, Eigen::VectorXd

static const char* kOutFile =
    "/global/homes/h/hyvchen/Regrid_Benchmark/src/GCR_Angle_test/results/angle_throughput_serial.csv";

// ---- timer ----
struct Timer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0;
    void start() { t0 = clock::now(); }
    double stop_s() const {
        auto t1 = clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        return dt.count();
    }
};

// ---- kernels (tiny; compiler will inline with -O3) ----
static inline double angle_arctan_crossdot(const V3_T<double>& u, const V3_T<double>& v) {
    const double cn = u.cross(v).norm();
    const double d  = u.dot(v);
    return std::atan(cn / d);   // quadrant-blind variant (kept to match prior label)
}
static inline double angle_kahan_atan(const V3_T<double>& u, const V3_T<double>& v) {
    // 2*atan(||u-v|| / ||u+v||) -- Kahan’s half-angle form
    double n_diff = (u - v).norm();
    double n_sum  = (u + v).norm();
    return 2.0 * std::atan(n_diff / n_sum);
}

// [To Do] Try to make the content the same and rerun it
static inline double angle_asin_cross(const V3_T<double>& u, const V3_T<double>& v) {
    // const double cn = u.cross(v).norm();//correct

    const double cn = u.dot(v); //Incorrect, just. for the testing purpose
    return std::asin(cn);
}
static inline double angle_acos_dot(const V3_T<double>& u, const V3_T<double>& v) {
    const double d = u.dot(v);

    //const double d = u.cross(v).norm();
    return std::acos(d);
}

// fixed unit vectors with tiny jitter later
static inline void fixed_vectors(V3_T<double>& u, V3_T<double>& v) {
    u << 5028374390644146 * std::ldexp(1.0, -53),
         -7472957205960756 * std::ldexp(1.0, -53),
          6254432438282003 * std::ldexp(1.0, -79);
    v << 5167685454902838 * std::ldexp(1.0, -53),
         -7377307466399399 * std::ldexp(1.0, -53),
          4525606513452550 * std::ldexp(1.0, -78);
    // normalize to be safe
    const double nu = u.norm(); if (nu != 0.0) u /= nu;
    const double nv = v.norm(); if (nv != 0.0) v /= nv;
}

// row for CSV
struct Row {
    const char* method;
    std::size_t N;
    double ns_per_angle;
    double angles_per_sec;
};

// run one kernel over prebuilt data (serial)
template <typename Kernel>
static Row bench_one(const char* name,
                     Kernel kernel,
                     const Eigen::MatrixXd& ptsU,
                     const Eigen::MatrixXd& ptsV,
                     Eigen::VectorXd& out)
{
    const std::size_t N = static_cast<std::size_t>(ptsU.rows());
    assert(ptsV.rows() == static_cast<long>(N));
    assert(out.size() == static_cast<long>(N));

    // warmup (small subset)
    {
        double wsum = 0.0;
        const std::size_t W = std::min<std::size_t>(N, 100000);
        for (std::size_t i = 0; i < W; ++i) {
            V3_T<double> u(ptsU(i,0), ptsU(i,1), ptsU(i,2));
            V3_T<double> v(ptsV(i,0), ptsV(i,1), ptsV(i,2));
            wsum += kernel(u, v);
        }
        if (wsum == 42.0) std::cerr << "(warmup) " << wsum << "\n";
    }

    // timed run
    Timer T; T.start();
    for (std::size_t i = 0; i < N; ++i) {
        V3_T<double> u(ptsU(i,0), ptsU(i,1), ptsU(i,2));
        V3_T<double> v(ptsV(i,0), ptsV(i,1), ptsV(i,2));
        out[i] = kernel(u, v);
    }
    const double secs = T.stop_s();

    volatile double checksum = out.sum(); // DCE guard
    if (checksum == std::numeric_limits<double>::infinity())
        std::cerr << "(ignore) " << checksum << "\n";

    Row r;
    r.method = name;
    r.N = N;
    r.ns_per_angle = (secs * 1e9) / double(N);
    r.angles_per_sec = double(N) / std::max(secs, 1e-18);
    return r;
}

int main(int argc, char** argv) {

    // Default logarithmic sizes; optionally override via CLI exponents:
    //   ./GCR_angle_perf_serial 2 7   -> N in {1e2, 1e3, ..., 1e7}
    int exp_lo = 2, exp_hi = 7;
    if (argc == 3) {
        exp_lo = std::atoi(argv[1]);
        exp_hi = std::atoi(argv[2]);
        if (exp_lo < 0 || exp_hi < exp_lo) {
            std::cerr << "Invalid exponents; expected: exp_lo exp_hi with 0 <= lo <= hi\n";
            return EXIT_FAILURE;
        }
    }

    std::vector<std::size_t> sizes;
    for (int e = exp_lo; e <= exp_hi; ++e) {
        std::size_t n = 1;
        for (int i = 0; i < e; ++i) n *= 10;
        sizes.push_back(n);
    }

    // Output rows
    std::vector<Row> rows;
    rows.reserve(sizes.size() * 4);

    // RNG for reproducible small jitters
    std::mt19937_64 rng(0xC0FFEEULL);
    std::normal_distribution<double> gauss(0.0, 1.0);

    for (std::size_t N : sizes) {
        std::cout << "[N=" << N << "] preparing data...\n";

        // Build inputs (N×3), output sink
        Eigen::MatrixXd ptsU(N, 3), ptsV(N, 3);
        Eigen::VectorXd out(N);

        // initialize with fixed vectors + tiny perturbations (scale fixed across N)
        V3_T<double> u0, v0; fixed_vectors(u0, v0);
        ptsU.rowwise() = u0.transpose();
        ptsV.rowwise() = v0.transpose();

        // add tiny IID jitter to avoid exact-duplicate cost-pathologies but keep near-unit
        // NOTE: keep perturbation scale small and constant across N to ensure fairness
        const double eps = 1e-12;
        for (std::size_t i = 0; i < N; ++i) {
            for (int c = 0; c < 3; ++c) {
                ptsU(i, c) += eps * gauss(rng);
                ptsV(i, c) += eps * gauss(rng);
            }
        }
        // (optional) renormalize rows to unit length, but skip to keep work identical across methods

        // Bench all methods on the exact same data (serial)
        rows.push_back(bench_one("arctan", angle_arctan_crossdot, ptsU, ptsV, out));
        rows.push_back(bench_one("atan2",  angle_kahan_atan,     ptsU, ptsV, out));
        rows.push_back(bench_one("asin",   angle_asin_cross,     ptsU, ptsV, out));
        rows.push_back(bench_one("acos",   angle_acos_dot,       ptsU, ptsV, out));

        std::cout << "  done N=" << N << "\n";
    }

    // write CSV
    std::ofstream fout(kOutFile, std::ios::trunc);
    if (!fout) {
        std::cerr << "ERROR: cannot open output file: " << kOutFile << "\n";
        return EXIT_FAILURE;
    }
    fout.setf(std::ios::fixed); fout.precision(6);
    fout << "method,N,ns_per_angle,angles_per_sec\n";
    for (const auto& r : rows) {
        fout << '"' << r.method << '"' << ','
             << r.N << ','
             << r.ns_per_angle << ','
             << r.angles_per_sec << '\n';
    }
    std::cout << "Wrote: " << kOutFile << "\n";
    return 0;
}
