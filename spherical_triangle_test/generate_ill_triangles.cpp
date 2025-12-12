// /global/homes/h/hyvchen/Regrid_Benchmark/src/spherical_triangle_test/generate_ill_triangles.cpp
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "mat_mesh_io.h"            // (not strictly needed to write CSV)
#include "regrid_benchmark_utils.h" // for Eigen aliases/utilities if you later expand

// ---- base-2 significand/exponent writer ------------------------------------
static inline std::pair<int64_t,int64_t> decompose_sigexp(double v) {
    if (v == 0.0) return {0, 0};
    int e = 0;
    double m = std::frexp(v, &e);                      // v = m * 2^e, m in [0.5,1) or (-1,-0.5]
    constexpr int p = std::numeric_limits<double>::digits; // 53
    double scaled = std::ldexp(m, p);                  // m * 2^p
    long long sig = std::llround(scaled);             // exact for finite double
    int64_t exp  = static_cast<int64_t>(e - p);
    const long long two_p = (p < 63) ? (1LL << p) : 0LL;
    if (two_p && sig == two_p) { sig >>= 1; ++exp; }  // renormalize if needed
    return {static_cast<int64_t>(sig), exp};
}

static inline void write_vertex_sigexp(std::ofstream& fout, const Eigen::Vector3d& v) {
    auto [xs, xe] = decompose_sigexp(v.x());
    auto [ys, ye] = decompose_sigexp(v.y());
    auto [zs, ze] = decompose_sigexp(v.z());
    fout << xs << ',' << xe << ',' << ys << ',' << ye << ',' << zs << ',' << ze;
}

// ---- random helpers ---------------------------------------------------------
static inline Eigen::Vector3d rand_unit_vec(std::mt19937_64& rng) {
    // Uniform on S^2 via z ~ U[-1,1], phi ~ U[0,2π]
    static thread_local std::uniform_real_distribution<double> U01(0.0, 1.0);
    double z   = 2.0*U01(rng) - 1.0;
    double phi = 2.0*M_PI * U01(rng);
    double r   = std::sqrt(std::max(0.0, 1.0 - z*z));
    return Eigen::Vector3d(r*std::cos(phi), r*std::sin(phi), z);
}

static inline void local_orthonormal_frame(const Eigen::Vector3d& w,
                                           Eigen::Vector3d& u,
                                           Eigen::Vector3d& v) {
    // Make u, v s.t. (u, v, w) is ONB. Avoid near-parallel reference axis.
    const Eigen::Vector3d ref = (std::fabs(w.z()) < 0.9) ? Eigen::Vector3d::UnitZ()
                                                         : Eigen::Vector3d::UnitX();
    u = ref.cross(w).normalized();
    v = w.cross(u); // already normalized if u,w are
}

int main(int argc, char** argv) {
    // --- parameters (same grids as before) ---
    int n = 60;    // number of polar angles (default)
    int m = 360;   // number of φ subdivisions (default)
    if (argc >= 2) n = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) m = std::max(2, std::atoi(argv[2])); // need at least 2 so (m-1) >= 1

    const double angle_n = 1.0;        // base colatitude (~57°) in radians
    const double angle_m = M_PI;       // sweep φ over [0, π]
    const double hm      = angle_m / static_cast<double>(m);
    const double ratio   = 0.8;        // geometric decay of colatitudes

    // geometric progression of colatitudes
    std::vector<double> anglens(static_cast<size_t>(n));
    anglens[0] = angle_n;
    for (int i = 1; i < n; ++i)
        anglens[static_cast<size_t>(i)] = anglens[static_cast<size_t>(i-1)] * ratio;

    // ---- output path (honor SPHTRI_OUT_DIR if set; don't create dirs) ----
    const char* out_env = std::getenv("SPHTRI_OUT_DIR");
    std::string out_dir = out_env && *out_env ? std::string(out_env)
                                              : "/global/homes/h/hyvchen/Regrid_Benchmark/src/spherical_triangle_test/output";
    if (!out_dir.empty() && out_dir.back() != '/' && out_dir.back() != '\\') out_dir.push_back('/');
    const std::string out_name = "ill_triangles_" + std::to_string(m) + "_" + std::to_string(n) + ".csv";
    const std::string out_path = out_dir + out_name;

    std::ofstream fout(out_path);
    if (!fout) {
        std::cerr << "ERROR: cannot open output file: " << out_path << "\n";
        return 1;
    }

    // header
    fout << "face_id,"
         << "v0x_sig,v0x_exp,v0y_sig,v0y_exp,v0z_sig,v0z_exp,"
         << "v1x_sig,v1x_exp,v1y_sig,v1y_exp,v1z_sig,v1z_exp,"
         << "v2x_sig,v2x_exp,v2y_sig,v2y_exp,v2z_sig,v2z_exp\n";

    // ---- random engine ----
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> Udelta(0.0, 1e-8); // tiny azimuth nudge δ ∈ [0,1e-8]

    // ---- main loop: i1 = 0..n-1; i2 = i1..n-1; φ = hm..π-hm ----
    std::uint64_t face_id = 0;

    for (int i1 = 0; i1 < n; ++i1) {
        const double theta1 = anglens[static_cast<size_t>(i1)];
        const double s1 = std::sin(theta1), c1 = std::cos(theta1);

        for (int i2 = i1; i2 < n; ++i2) {
            const double theta2 = anglens[static_cast<size_t>(i2)];
            const double s2 = std::sin(theta2), c2 = std::cos(theta2);

            for (int j = 1; j <= m - 1; ++j) {
                const double phi = hm * static_cast<double>(j);

                // --- per-face random anchor and local frame ---
                const Eigen::Vector3d w = rand_unit_vec(rng); // a = w
                Eigen::Vector3d u, v;
                local_orthonormal_frame(w, u, v);

                // --- tiny random azimuth δ for b so it’s not in a coordinate plane ---
                const double delta = Udelta(rng);
                const Eigen::Vector3d dir_b = std::cos(delta)*u + std::sin(delta)*v;
                const Eigen::Vector3d b = s1 * dir_b + c1 * w;

                // --- c uses (θ2, φ) in the SAME local frame (u,v,w) ---
                const Eigen::Vector3d dir_c = std::cos(phi)*u + std::sin(phi)*v;
                const Eigen::Vector3d c = s2 * dir_c + c2 * w;

                // a is exactly w
                const Eigen::Vector3d a = w;

                // CSV row
                fout << face_id << ',';
                write_vertex_sigexp(fout, a); fout << ',';
                write_vertex_sigexp(fout, b); fout << ',';
                write_vertex_sigexp(fout, c); fout << '\n';

                ++face_id;
            }
        }
    }

    fout.flush();
    if (!fout) {
        std::cerr << "ERROR: failed while writing CSV: " << out_path << "\n";
        return 2;
    }

    const std::uint64_t expected = static_cast<std::uint64_t>(m - 1)
                                 * static_cast<std::uint64_t>(n)
                                 * static_cast<std::uint64_t>(n + 1) / 2ULL;

    std::cout << "Wrote " << face_id << " triangles to: " << out_path << "\n"
              << "Expected count: " << expected << "\n";
    return 0;
}
