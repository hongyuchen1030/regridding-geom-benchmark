// /global/homes/h/hyvchen/Regrid_Benchmark/src/GCR_Angle_test/GCR_generate_arcs.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>
#include "regrid_benchmark_utils.h"

static inline void create_directory(const char* path) {
    if (mkdir(path, 0755) != 0) {
        if (errno != EEXIST) {
            std::cerr << "Error creating directory '" << path << "': "
                      << std::strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv) {
    // Usage: ./GCR_generate_arcs [seed] [total] [ill]
    std::uint64_t seed = 42;
    std::size_t   total = 10000000;
    std::size_t   ill   = 100000;

    if (argc >= 2) seed  = static_cast<std::uint64_t>(std::stoull(argv[1]));
    if (argc >= 3) total = static_cast<std::size_t>(std::stoull(argv[2]));
    if (argc >= 4) ill   = static_cast<std::size_t>(std::stoull(argv[3]));

    const char* results_dir = "../results";
    create_directory(results_dir);

    // Generate using your existing utility; type is VecPair<double>
    std::vector<VecPair<double>> arcs =
        generate_pairs_0_90_with_ill_coverage<double>(seed, total, ill);

    // Ensure ascending by angle (generator may already do it; be safe)
    std::sort(arcs.begin(), arcs.end(),
              [](const VecPair<double>& a, const VecPair<double>& b) {
                  return a.angle_deg < b.angle_deg;
              });

    // Write generated_arcs.csv
    const std::string csv_path = std::string(results_dir) + "/generated_arcs.csv";
    {
        struct stat sb{};
        if (stat(csv_path.c_str(), &sb) == 0)
            std::cout << "Overwriting existing file: '" << csv_path << "'\n";
        else
            std::cout << "Creating new file: '" << csv_path << "'\n";
    }

    std::ofstream csv(csv_path, std::ios::trunc);
    if (!csv) {
        std::cerr << "Failed to open output file: " << csv_path << "\n";
        return 1;
    }

    csv << "ref_degree,"
           "x1_significand,x1_exponent,"
           "y1_significand,y1_exponent,"
           "z1_significand,z1_exponent,"
           "x2_significand,x2_exponent,"
           "y2_significand,y2_exponent,"
           "z2_significand,z2_exponent\n";

    for (const auto& p : arcs) {
        csv << p.angle_deg << ","
            << DecomposedFloat(p.u[0]).toCSV() << ","
            << DecomposedFloat(p.u[1]).toCSV() << ","
            << DecomposedFloat(p.u[2]).toCSV() << ","
            << DecomposedFloat(p.v[0]).toCSV() << ","
            << DecomposedFloat(p.v[1]).toCSV() << ","
            << DecomposedFloat(p.v[2]).toCSV() << "\n";
    }

    std::cout << "Generated " << arcs.size()
              << " arcs and wrote " << csv_path << "\n";
    return 0;
}
