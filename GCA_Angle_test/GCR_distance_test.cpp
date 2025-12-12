#include <iostream>
#include <array>
#include <cmath>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include "regrid_benchmark_utils.h"

// Helper: create directory with POSIX mkdir
inline void create_directory(const char* path) {
    if (mkdir(path, 0755) != 0) {
        if (errno != EEXIST) {
            std::cerr << "Error creating directory '" << path << "': "
                      << std::strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

double angle_arctan_crossdot(const V3_T<double>& u, const V3_T<double>& v) {
    double cross_norm = u.cross(v).norm();
    double dot = u.dot(v);
    return std::atan2(cross_norm, dot);
}

double angle_kahan(const V3_T<double>& u, const V3_T<double>& v) {
    double n_diff = (u - v).norm();
    double n_sum  = (u + v).norm();
    return 2.0 * std::atan2(n_diff, n_sum); 
}

double angle_yac_asin(const V3_T<double>& u, const V3_T<double>& v) {
    return std::asin(u.cross(v).norm());
}

double angle_yac_acos(const V3_T<double>& u, const V3_T<double>& v) {
    return std::acos(u.dot(v));
}


int main(int argc, char** argv) {
    // Fixed locations per your request
    const char* results_dir = "/global/homes/h/hyvchen/Regrid_Benchmark/src/GCR_Angle_test/results";
    const char* input_csv   = "/global/homes/h/hyvchen/Regrid_Benchmark/src/GCR_Angle_test/results/generated_arcs.csv";

    // Ensure results directory exists
    create_directory(results_dir);

    // ---- 1) Load arcs from CSV (uses VecPair<double> and DecomposedFloat::toFloat) ----
    std::vector<VecPair<double>> arcs;
    if (!load_arcs_csv<double>(input_csv, arcs)) {
        std::cerr << "ERROR: failed to load arcs from: " << input_csv << "\n";
        return EXIT_FAILURE;
    }
    if (arcs.empty()) {
        std::cerr << "ERROR: no arcs loaded from: " << input_csv << "\n";
        return EXIT_FAILURE;
    }

    // Keep a clean ascending order by angle (CSV should already be sorted; be safe)
    std::sort(arcs.begin(), arcs.end(),
              [](const VecPair<double>& a, const VecPair<double>& b) {
                  return a.angle_deg < b.angle_deg;
              });

    // ---- 2) Compute angles via four methods and write angle_results.csv ----
    {
        std::string csv_path = std::string(results_dir) + "/angle_results.csv";
        struct stat sb{};
        if (stat(csv_path.c_str(), &sb) == 0)
            std::cout << "Overwriting existing file: '" << csv_path << "'\n";
        else
            std::cout << "Creating new file: '" << csv_path << "'\n";

        std::ofstream csv(csv_path, std::ios::trunc);
        if (!csv) {
            std::cerr << "ERROR: cannot open output file: " << csv_path << "\n";
            return EXIT_FAILURE;
        }

        csv << "input_deg,"
               "arctan_significand,arctan_exponent,"
               "kahan_significand,kahan_exponent,"
               "asin_significand,asin_exponent,"
               "acos_significand,acos_exponent\n";

        for (const auto& p : arcs) {
            const V3_T<double>& u = p.u;
            const V3_T<double>& v = p.v;

            double a_arctan = angle_arctan_crossdot(u, v);
            double a_kahan  = angle_kahan(u, v);
            double a_asin   = angle_yac_asin(u, v);
            double a_acos   = angle_yac_acos(u, v);

            csv << p.angle_deg << ","
                << DecomposedFloat(a_arctan).toCSV() << ","
                << DecomposedFloat(a_kahan ).toCSV() << ","
                << DecomposedFloat(a_asin  ).toCSV() << ","
                << DecomposedFloat(a_acos  ).toCSV() << "\n";
        }
    }

    std::cout << "Read " << arcs.size()
              << " arcs from " << input_csv
              << " and wrote method outputs to " << results_dir << "/angle_results.csv\n";
    return 0;
}
