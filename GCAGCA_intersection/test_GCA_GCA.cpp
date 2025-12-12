#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cerrno>

#include "../regrid_benchmark_utils.h"

using V3d = V3_T<double>;
using operator_helper::to_arr;
using operator_helper::from_arr_v3t;
using operator_helper::cross_fma;
using operator_helper::SumOfSquaresC;
using operator_helper::acc_sqrt_re;
using operator_helper::taylor_normalization; // from utils
using operator_helper::two_sum;

// EFT helpers
namespace {
  using operator_helper::two_prod_fma;
  using operator_helper::fast_two_sum;
  using operator_helper::CompDotC;

  // (a*b) - (c*d) with residual (hi, lo)
  static inline std::tuple<double,double>
  fmms_re(double a, double b, double c, double d) {
    auto [p1, s1] = two_prod_fma(a, b);
    auto [h2, r2] = two_prod_fma(c, -d);
    auto [p2, q2] = two_sum(p1, h2);
    double s2 = s1 + (q2 + r2);
    return {p2, s2}; // (hi, lo)
  }

  // First-level compensated cross: returns {xh,xl, yh,yl, zh,zl}
  static inline std::array<double,6>
  cross_comp_first(const V3d& x1, const V3d& x2) {
    auto [nx, enx] = fmms_re(x1[1], x2[2], x1[2], x2[1]);
    auto [ny, eny] = fmms_re(x1[2], x2[0], x1[0], x2[2]);
    auto [nz, enz] = fmms_re(x1[0], x2[1], x1[1], x2[0]);
    return {nx,enx, ny,eny, nz,enz};
  }

  // Second-level compensated cross of two compensated vectors:
  // Build the 6-term CompDotC expansion per component (AccuCross pattern).
  static inline std::array<double,6>
  cross_comp_second(const std::array<double,6>& n1,
                    const std::array<double,6>& n2) {
    std::array<double, 6> n3;

    //Decompose n1, n2 into two arrays of 3 elements each
    std::array<double, 3> n1_v = {n1[0], n1[2], n1[4]};
    std::array<double, 3> n1_e = {n1[1], n1[3], n1[5]};
    std::array<double, 3> n2_v = {n2[0], n2[2], n2[4]};
    std::array<double, 3> n2_e = {n2[1], n2[3], n2[5]};

    double n1x = n1_v[0];
    double en1x = n1_e[0];
    double n1y = n1_v[1];
    double en1y = n1_e[1];
    double n1z = n1_v[2];
    double en1z = n1_e[2];

    double n2x = n2_v[0];
    double en2x = n2_e[0];
    double n2y = n2_v[1];
    double en2y = n2_e[1];
    double n2z = n2_v[2];
    double en2z = n2_e[2];

    // Calculate n3[0] component
    //-en1z en2y + en1y en2z + en2z n1y - en2y n1z - en1z n2y - n1z n2y + en1y n2z + n1y n2z
    {
        double s, c;
        std::tie(s, c) =  CompDotC<double,8>(
                {-en1z, en1y, n1y,  -n1z,  -n2y, -n1z, en1y, n1y},
                {en2y,  en2z, en2z, en2y, en1z,   n2y, n2z, n2z}
        );
        n3[0] = s;
        n3[1] = c;
    }


    // Calculate n3[1] component
    //en1z en2x - en1x en2z - en2z n1x + en2x n1z + en1z n2x + n1z n2x - en1x n2z - n1x n2z
    {
        double s, c;
        std::tie(s, c) =  CompDotC<double,8>(
                {en1z, -en1x, -n1x,   n1z, n2x,   n1z, -en1x, -n1x},
                {en2x,  en2z,  en2z, en2x, en1z,  n2x,  n2z,  n2z}
        );
        n3[2] = s;
        n3[3] = c;
    }

    // Calculate n3[2] component
    // -en1y en2x + en1x en2y + en2y n1x - en2x n1y - en1y n2x - n1y n2x + en1x n2y + n1x n2y
    {
        double s, c;
        std::tie(s, c) =  CompDotC<double,8>(
                {-en1y, en1x, n1x,  -n1y, -n2x, -n1y, en1x, n1x},
                {en2x,  en2y, en2y, en2x, en1y, n2x,  n2y,  n2y}
        );
        n3[4] = s;
        n3[5] = c;
    }
    return n3;
  }

  // Euclidean norm (N,n) of a compensated vector n3={xh,xl,yh,yl,zh,zl}
  static inline std::tuple<double,double>
  comp_norm(const std::array<double,6>& n3) {
    std::array<double,3> vh = { n3[0], n3[2], n3[4] };
    std::array<double,3> vl = { n3[1], n3[3], n3[5] };
    double S,s; std::tie(S,s) = SumOfSquaresC<double,3>(vh, vl);
    return acc_sqrt_re(S, s); // (N, n)
  }
} // namespace

// -------------------- CSV loader --------------------
struct PairRow {
  std::uint64_t pairs_id;
  double ref_deg;
  V3d a0, a1, b0, b1;
};

static bool parse_sigexp(std::istringstream& ss, double& out) {
  std::string s_sig, s_exp;
  if (!std::getline(ss, s_sig, ',')) return false;
  if (!std::getline(ss, s_exp, ',')) return false;
  errno = 0;
  long long isig = std::strtoll(s_sig.c_str(), nullptr, 10);
  long long iexp = std::strtoll(s_exp.c_str(), nullptr, 10);
  if (errno != 0) return false;
  DecomposedFloat df(static_cast<int64_t>(isig), static_cast<int64_t>(iexp));
  out = df.toFloat<double>();
  return true;
}

static bool load_pairs_csv(const std::string& path, std::vector<PairRow>& out) {
  std::ifstream fin(path);
  if (!fin) {
    std::cerr << "Failed to open input CSV: " << path << "\n";
    return false;
  }
  std::string line;
  if (!std::getline(fin, line)) {
    std::cerr << "Empty file or header missing: " << path << "\n";
    return false;
  }
  out.clear();
  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string tok;
    PairRow row{};

    // pairs_id
    if (!std::getline(ss, tok, ',')) break;
    row.pairs_id = static_cast<std::uint64_t>(std::strtoull(tok.c_str(), nullptr, 10));

    // ref_angle_deg
    if (!std::getline(ss, tok, ',')) break;
    row.ref_deg = std::strtod(tok.c_str(), nullptr);

    // a0
    if (!parse_sigexp(ss, row.a0[0])) break;
    if (!parse_sigexp(ss, row.a0[1])) break;
    if (!parse_sigexp(ss, row.a0[2])) break;
    // a1
    if (!parse_sigexp(ss, row.a1[0])) break;
    if (!parse_sigexp(ss, row.a1[1])) break;
    if (!parse_sigexp(ss, row.a1[2])) break;
    // b0
    if (!parse_sigexp(ss, row.b0[0])) break;
    if (!parse_sigexp(ss, row.b0[1])) break;
    if (!parse_sigexp(ss, row.b0[2])) break;
    // b1
    if (!parse_sigexp(ss, row.b1[0])) break;
    if (!parse_sigexp(ss, row.b1[1])) break;
    if (!parse_sigexp(ss, row.b1[2])) break;

    out.push_back(std::move(row));
  }
  if (out.empty()) {
    std::cerr << "No data rows loaded from: " << path << "\n";
    return false;
  }
  return true;
}


static inline V3d cross_direct(const V3d& a, const V3d& b) {
  return V3d(a.y()*b.z() - a.z()*b.y(),
             a.z()*b.x() - a.x()*b.z(),
             a.x()*b.y() - a.y()*b.x());
}

static inline V3d cross_kahan(const V3d& a, const V3d& b) {
  auto aa = to_arr(a);
  auto bb = to_arr(b);
  auto cc = cross_fma<double>(aa, bb);
  return from_arr_v3t<double>(cc);
}

// Normalize (direct) with guard; return 0 if ok, 1 if degenerate
static inline int normalize_direct(V3d& v) {
  double n2 = v.squaredNorm();
  if (!(n2 > 0.0) || !std::isfinite(n2)) return 1;
  v /= std::sqrt(n2);
  return 0;
}



// -------------------- Methods --------------------
static void method_direct(const PairRow& r, double& vx, double& vy, double& vz) {
  V3d n1 = cross_direct(r.a0, r.a1);
  V3d n2 = cross_direct(r.b0, r.b1);
  V3d v  = cross_direct(n1, n2);
  if (normalize_direct(v) != 0) { vx = vy = vz = 0.0; return; }
  vx = v[0]; vy = v[1]; vz = v[2];
}

static void method_kahan(const PairRow& r, double& vx, double& vy, double& vz) {
  V3d n1 = cross_kahan(r.a0, r.a1);
  V3d n2 = cross_kahan(r.b0, r.b1);
  V3d v  = cross_kahan(n1, n2);
  if (normalize_direct(v) != 0) { vx = vy = vz = 0.0; return; }
  vx = v[0]; vy = v[1]; vz = v[2];
}

// Normalize (EFT): given hi/lo parts (v,e), compute |v| via SumOfSquaresC and acc_sqrt_re
static inline int normalize_eft(std::array<double,3>& v,
                                std::array<double,3>& e)
{
  auto SoS = SumOfSquaresC<double,3>(v, e);  // (S, s) for |v|^2
  double N, n; std::tie(N, n) = acc_sqrt_re(std::get<0>(SoS), std::get<1>(SoS));
  if (!(N > 0.0) || !std::isfinite(N)) return 1;
  for (int i = 0; i < 3; ++i) {
    double S = v[i], s = e[i];
    double S1, s1; std::tie(S1, s1) = two_sum(S, s);
    v[i] = S1 / N ;//+ ((N * s1 - n * S1) / (N * N));
    e[i] = 0.0;
  }
  return 0;
}

// ===== EFT ==================
static void method_eft(const PairRow& r, double& vx, double& vy, double& vz) {
  // First crosses with AccuCross on (a0 × a1) and (b0 × b1) — zero-error overload
  V3d n1h, n1l, n2h, n2l;
  {
    auto hl1 = operator_helper::AccuCross(r.a0, r.a1);
    n1h = hl1.first; n1l = hl1.second;
    auto hl2 = operator_helper::AccuCross(r.b0, r.b1);
    n2h = hl2.first; n2l = hl2.second;
  }
  // Second cross: full error-aware AccuCross with value/error pairs
  auto vhl = operator_helper::AccuCross(n1h, n1l, n2h, n2l);
  std::array<double,3> vh = to_arr(vhl.first);
  std::array<double,3> vl = to_arr(vhl.second);
  if (normalize_eft(vh, vl) != 0) { vx = vy = vz = 0.0; return; }
  vx = vh[0] + vl[0];
  vy = vh[1] + vl[1];
  vz = vh[2] + vl[2];
}

/
static void usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --in <pairs.csv> --out <results.csv> [--method all|direct|kahan|eft]\n";
}

int main(int argc, char** argv) {
  std::string in_csv, out_csv, method = "all";
  for (int i=1; i<argc; ++i) {
    std::string a = argv[i];
    auto need = [&](int n=1){ return (i+n < argc); };
    if (a == "--in" && need())       in_csv = argv[++i];
    else if (a == "--out" && need()) out_csv = argv[++i];
    else if (a == "--method" && need()) method = argv[++i]; // accepted but ignored
    else { usage(argv[0]); return 1; }
  }
  if (in_csv.empty() || out_csv.empty()) { usage(argv[0]); return 1; }

  std::vector<PairRow> rows;
  if (!load_pairs_csv(in_csv, rows)) return 1;

  std::ofstream out(out_csv, std::ios::trunc);
  if (!out) {
    std::cerr << "Failed to open output: " << out_csv << "\n";
    return 1;
  }

  // Header with all method columns (sig/exp triples)
  out << "pairs_id,ref_angle_deg,"
         "direct_vx_sig,direct_vx_exp,direct_vy_sig,direct_vy_exp,direct_vz_sig,direct_vz_exp,"
         "kahan_vx_sig,kahan_vx_exp,kahan_vy_sig,kahan_vy_exp,kahan_vz_sig,kahan_vz_exp,"
         "eft_vx_sig,eft_vx_exp,eft_vy_sig,eft_vy_exp,eft_vz_sig,eft_vz_exp\n";

  // Always compute all three methods
  for (const auto& r : rows) {
    double dvx=0, dvy=0, dvz=0;
    double kvx=0, kvy=0, kvz=0;
    double evx=0, evy=0, evz=0;

    method_direct(r, dvx, dvy, dvz);
    method_kahan (r, kvx, kvy, kvz);
    method_eft   (r, evx, evy, evz);

    DecomposedFloat ddx(dvx), ddy(dvy), ddz(dvz);
    DecomposedFloat kdx(kvx), kdy(kvy), kdz(kvz);
    DecomposedFloat edx(evx), edy(evy), edz(evz);

    out << r.pairs_id << "," << r.ref_deg << ","
        << ddx.toCSV() << "," << ddy.toCSV() << "," << ddz.toCSV() << ","
        << kdx.toCSV() << "," << kdy.toCSV() << "," << kdz.toCSV() << ","
        << edx.toCSV() << "," << edy.toCSV() << "," << edz.toCSV() << "\n";
  }

  std::cout << "Wrote results: " << out_csv
            << " (" << rows.size() << " pairs, all three methods per row)\n";
  return 0;
}
