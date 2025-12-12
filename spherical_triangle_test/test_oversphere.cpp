#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iomanip>

#include <Eigen/Core>

#include "arpist_quad.h"
#include "tempest_quad.h"
#include "yac_quad.h"
#include "eriksson_area.h"
#include "../regrid_benchmark_utils.h"  // DecomposedFloat, operator_helper, etc.

// ---------- mesh container ----------
struct UnstructuredGrid {
  std::vector<Eigen::Vector3d> xs;     // n x 3 (unit sphere)
  std::vector<Eigen::Vector3i> elems;  // f x 3 (0-based)
  std::vector<int> face_ids;           // face id per triangle
};

static inline void die_if(bool bad, const std::string& msg) {
  if (bad) throw std::runtime_error(msg);
}

static inline double df_to_double(int64_t sig, int64_t exp) {
  DecomposedFloat df(sig, exp);
  return df.toFloat<double>();
}

// CSV format (19 columns):
// face_id,
// v0x_sig,v0x_exp, v0y_sig,v0y_exp, v0z_sig,v0z_exp,
// v1x_sig,v1x_exp, v1y_sig,v1y_exp, v1z_sig,v1z_exp,
// v2x_sig,v2x_exp, v2y_sig,v2x_exp, v2z_sig,v2z_exp
static UnstructuredGrid load_mesh_from_csv(const std::string& csv_path) {
  std::ifstream fin(csv_path);
  if (!fin) throw std::runtime_error("cannot open CSV: " + csv_path);

  std::string line;
  if (!std::getline(fin, line))
    throw std::runtime_error("empty CSV: " + csv_path);

  UnstructuredGrid g;

  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    std::vector<std::string> t; t.reserve(32);
    std::stringstream ss(line);
    std::string it;
    while (std::getline(ss, it, ',')) t.push_back(it);
    if (t.size() != 19) {
      std::ostringstream msg;
      msg << "CSV row has " << t.size() << " columns (expect 19): " << line;
      throw std::runtime_error(msg.str());
    }

    auto as_ll = [](const std::string& s)->int64_t {
      size_t pos = 0; return std::stoll(s, &pos, 10);
    };

    const int face_id = static_cast<int>(as_ll(t[0]));

    auto V = [&](int sig_idx, int exp_idx)->double {
      return df_to_double(as_ll(t[sig_idx]), as_ll(t[exp_idx]));
    };

    Eigen::Vector3d v0, v1, v2;
    v0.x() = V( 1,  2); v0.y() = V( 3,  4); v0.z() = V( 5,  6);
    v1.x() = V( 7,  8); v1.y() = V( 9, 10); v1.z() = V(11, 12);
    v2.x() = V(13, 14); v2.y() = V(15, 16); v2.z() = V(17, 18);

    // Face-disjoint vertices
    const int base = static_cast<int>(g.xs.size());
    g.xs.push_back(v0); g.xs.push_back(v1); g.xs.push_back(v2);
    g.elems.emplace_back(base + 0, base + 1, base + 2);
    g.face_ids.push_back(face_id);
  }

  if (g.elems.empty())
    throw std::runtime_error("no faces parsed from CSV: " + csv_path);

  // Normalize to unit sphere (defensive)
  for (auto& v : g.xs) {
    const double nrm = v.norm();
    if (nrm > 0) v /= nrm;
  }
  return g;
}

static inline double env_or(const char* name, double fallback) {
  if (const char* s = std::getenv(name)) {
    try { return std::stod(s); } catch (...) {}
  }
  return fallback;
}

// Write per-face method areas as sig/exp pairs (two columns per method)
// Methods recorded:
//   - ARPIST: deg=4, deg=8, adaptive
//   - Tempest: deg=4, deg=8, adaptive
//   - YAC: FMA only
static void write_methods_csv(
  const std::string& out_path,
  const std::vector<int>& face_ids,
  const std::vector<double>& A_arpist_4,
  const std::vector<double>& A_arpist_8,
  const std::vector<double>& A_arpist_adaptive,
  const std::vector<double>& A_tempest_4,
  const std::vector<double>& A_tempest_8,
  const std::vector<double>& A_tempest_adaptive,
  const std::vector<double>& A_yac_fma,
  const std::vector<double>& A_eriksson_fma)
{
  const size_t n = face_ids.size();
  if (A_arpist_4.size()!=n || A_arpist_8.size()!=n || A_arpist_adaptive.size()!=n ||
      A_tempest_4.size()!=n || A_tempest_8.size()!=n || A_tempest_adaptive.size()!=n ||
      A_yac_fma.size()!=n || A_eriksson_fma.size()!=n) {
    throw std::runtime_error("size mismatch among per-face area vectors");
  }


  std::ofstream fout(out_path);
  if (!fout) throw std::runtime_error("cannot open for write: " + out_path);

  // Header per your exact spec
  fout
    << "face_id,"
    << "arpist_4_sig,arpist_4_exp,"
    << "arpist_8_sig,arpist_8_exp,"
    << "arpist_adaptive_sig,arpist_adaptive_exp,"
    << "tempest_4_sig,tempest_4_exp,"
    << "tempest_8_sig,tempest_8_exp,"
    << "tempest_adaptive_sig,tempest_adaptive_exp,"
    << "yac_fma_sig,yac_fma_exp,"
    << "eriksson_fma_sig,eriksson_fma_exp\n";  


  for (size_t i = 0; i < n; ++i) {
    DecomposedFloat d_a4 (A_arpist_4[i]);
    DecomposedFloat d_a8 (A_arpist_8[i]);
    DecomposedFloat d_aa (A_arpist_adaptive[i]);
    DecomposedFloat d_t4 (A_tempest_4[i]);
    DecomposedFloat d_t8 (A_tempest_8[i]);
    DecomposedFloat d_ta (A_tempest_adaptive[i]);
    DecomposedFloat d_yf (A_yac_fma[i]);
    DecomposedFloat d_ef (A_eriksson_fma[i]); 

    fout << face_ids[i] << ","
         << d_a4.significand << "," << d_a4.exponent << ","
         << d_a8.significand << "," << d_a8.exponent << ","
         << d_aa.significand << "," << d_aa.exponent << ","
         << d_t4.significand << "," << d_t4.exponent << ","
         << d_t8.significand << "," << d_t8.exponent << ","
         << d_ta.significand << "," << d_ta.exponent << ","
         << d_yf.significand << "," << d_yf.exponent << ","
         << d_ef.significand << "," << d_ef.exponent      // <— NEW
         << "\n";
  }
  fout.close();
}

// --- helpers to build methods_area_<basename>.csv ---
static std::string basename_noext(const std::string& p) {
  const auto slash = p.find_last_of("/\\");
  std::string base = (slash == std::string::npos) ? p : p.substr(slash + 1);
  const auto dot = base.find_last_of('.');
  if (dot != std::string::npos) base.erase(dot);
  return base;
}

int main(int argc, char** argv) {
  try {
    // ---- Require full CSV path, no filename assumptions ----
    if (argc < 2) {
      std::cerr << "usage: " << argv[0] << " <input-mesh.csv>\n";
      return 1;
    }
    const std::string csv_path = argv[1];

    // Load CSV mesh
    auto g = load_mesh_from_csv(csv_path);
    std::cout << "[OK] CSV parsed: faces=" << g.elems.size()
              << " vertices=" << g.xs.size() << "\n";

    // ---- ARPIST per-face areas (deg=4,8,adaptive) ----
    auto area_arpist_deg = [&](int deg)->std::vector<double> {
      return arpist::spherical_integration(
        g.xs, g.elems,
        [](const Eigen::Vector3d&) { return 1.0; },
        deg);
    };

    const std::vector<double> A_arpist_4  = area_arpist_deg(4);
    const std::vector<double> A_arpist_8  = area_arpist_deg(8);
    const std::vector<double> A_arpist_ad = area_arpist_deg(-1); // adaptive

    // ---- Tempest per-face areas (deg=4,8,adaptive) ----
    std::vector<double> A_tempest_4(g.elems.size());
    std::vector<double> A_tempest_8(g.elems.size());
    std::vector<double> A_tempest_ad(g.elems.size());

    for (size_t f = 0; f < g.elems.size(); ++f) {
      const auto tri = g.elems[f];
      const Eigen::Vector3d vA = g.xs[tri[0]];
      const Eigen::Vector3d vB = g.xs[tri[1]];
      const Eigen::Vector3d vC = g.xs[tri[2]];

      // Use the split quadrature for fixed degrees; method call for adaptive
      A_tempest_4[f] = tempest::CalculateTriangleAreaTriQuadrature(vA, vB, vC, 4);
      A_tempest_8[f] = tempest::CalculateTriangleAreaTriQuadrature(vA, vB, vC, 8);
      A_tempest_ad[f] = tempest::CalculateTriangleAreaTriQuadratureMethod(vA, vB, vC);
    }

    // ---- YAC per-face areas (FMA only) ----
    std::vector<double> A_yac_f(g.elems.size());  // yac_fma
    for (size_t f = 0; f < g.elems.size(); ++f) {
      const auto t = g.elems[f];
      A_yac_f[f] = yac::TriangleAreaFMA(g.xs[t[0]], g.xs[t[1]], g.xs[t[2]]);
    }

    // ---- Eriksson per-face areas (FMA cross; spherical excess) ----
    std::vector<double> A_eriksson_f(g.elems.size());  // eriksson_fma
    for (size_t f = 0; f < g.elems.size(); ++f) {
      const auto t = g.elems[f];
      A_eriksson_f[f] = eriksson::TriangleAreaFMA(g.xs[t[0]], g.xs[t[1]], g.xs[t[2]]);
    }


    // ---- Output = methods_area_<input-file-basename>.csv
    // Default directory = alongside input CSV. Can override via env SPHTRI_OUT_DIR.
    std::string out_path;
    {
      const auto dir_end = csv_path.find_last_of("/\\");
      const std::string in_dir = (dir_end == std::string::npos) ? std::string("") : csv_path.substr(0, dir_end + 1);
      const std::string base_noext = basename_noext(csv_path); // strip extension

      const char* out_env = std::getenv("SPHTRI_OUT_DIR");
      const std::string out_dir = (out_env && *out_env) ? std::string(out_env) : in_dir;

      out_path = out_dir + "methods_area_" + base_noext + ".csv";
      if (!out_dir.empty() && out_dir.back() != '/' && out_dir.back() != '\\') {
        out_path = out_dir + "/" + "methods_area_" + base_noext + ".csv";
      }
    }

    write_methods_csv(out_path, g.face_ids,
                      A_arpist_4, A_arpist_8, A_arpist_ad,
                      A_tempest_4, A_tempest_8, A_tempest_ad,
                      A_yac_f,
                      A_eriksson_f); 

    std::cout << "Wrote per-face areas to: " << out_path << "\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
