// /global/homes/h/hyvchen/Regrid_Benchmark/src/spherical_triangle_test/decompose_mesh_vertices.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

#include "../regrid_benchmark_utils.h"

namespace {

/** Trim helper */
inline std::string trim(std::string s) {
    auto notspace = [](int ch){ return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    return s;
}

/** Split a CSV line by ',' without fancy quoting (our files are simple) */
static std::vector<std::string> split_csv_simple(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    std::istringstream ss(line);
    while (std::getline(ss, field, ',')) out.push_back(trim(field));
    return out;
}

/** Build output path: input "foo.csv" -> "foo_sigexp.csv" in same directory */
static std::string make_out_path(const std::string& in_path) {
    // find last dot
    auto pos = in_path.find_last_of('.');
    if (pos == std::string::npos) return in_path + "_sigexp.csv";
    return in_path.substr(0, pos) + "_sigexp.csv";
}

/** Process one mesh CSV */
static void process_one_csv(const std::string& in_csv) {
    std::ifstream fin(in_csv);
    if (!fin) {
        throw std::runtime_error("Failed to open input CSV: " + in_csv);
    }

    std::string out_csv = make_out_path(in_csv);
    std::ofstream fout(out_csv);
    if (!fout) {
        throw std::runtime_error("Failed to open output CSV: " + out_csv);
    }

    // Write header (19 columns)
    fout
        << "face_id,"
        << "v0x_sig,v0x_exp,v0y_sig,v0y_exp,v0z_sig,v0z_exp,"
        << "v1x_sig,v1x_exp,v1y_sig,v1y_exp,v1z_sig,v1z_exp,"
        << "v2x_sig,v2x_exp,v2y_sig,v2y_exp,v2z_sig,v2z_exp\n";

    std::string line;
    bool first = true;
    std::size_t nrows = 0;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        // Skip header if present
        if (first) {
            first = false;
            auto maybe = split_csv_simple(line);
            if (!maybe.empty() && maybe[0] == "face_id") {
                // header line, skip
                continue;
            }
            // else: fall through to parse this as data
        }

        auto tok = split_csv_simple(line);
        if (tok.size() != 10) {
            std::ostringstream oss;
            oss << "Unexpected token count (" << tok.size()
                << ") in line: " << line;
            throw std::runtime_error(oss.str());
        }

        // Parse face_id
        long long face_id_ll = 0;
        try {
            face_id_ll = std::stoll(tok[0]);
        } catch (...) {
            throw std::runtime_error("Failed to parse face_id in line: " + line);
        }

        // Parse 9 doubles (x1,y1,z1,x2,y2,z2,x3,y3,z3)
        double coords[9];
        for (int i = 0; i < 9; ++i) {
            try {
                coords[i] = std::stod(tok[1 + i]);
            } catch (...) {
                std::ostringstream oss;
                oss << "Failed to parse coordinate index " << i
                    << " in line: " << line;
                throw std::runtime_error(oss.str());
            }
        }

        // Decompose each double via DecomposedFloat<double>
        // Order: v0(x,y,z), v1(x,y,z), v2(x,y,z)
        DecomposedFloat d[9] = {
            DecomposedFloat(coords[0]), DecomposedFloat(coords[1]), DecomposedFloat(coords[2]),
            DecomposedFloat(coords[3]), DecomposedFloat(coords[4]), DecomposedFloat(coords[5]),
            DecomposedFloat(coords[6]), DecomposedFloat(coords[7]), DecomposedFloat(coords[8])
        };

        // Emit a row in the requested 19-column format
        fout << face_id_ll << ','
             << d[0].significand << ',' << d[0].exponent << ','
             << d[1].significand << ',' << d[1].exponent << ','
             << d[2].significand << ',' << d[2].exponent << ','
             << d[3].significand << ',' << d[3].exponent << ','
             << d[4].significand << ',' << d[4].exponent << ','
             << d[5].significand << ',' << d[5].exponent << ','
             << d[6].significand << ',' << d[6].exponent << ','
             << d[7].significand << ',' << d[7].exponent << ','
             << d[8].significand << ',' << d[8].exponent
             << "\n";

        ++nrows;
    }

    std::cerr << "[OK] Wrote " << nrows << " rows to: " << out_csv << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> inputs;

        if (argc > 1) {
            // Use provided paths
            for (int i = 1; i < argc; ++i) inputs.emplace_back(argv[i]);
        } else {
            // Default to your four meshes under pscratch
            const char* base = "/pscratch/sd/h/hyvchen/ico_meshes/";
            inputs = {
                std::string(base) + "ico_r7_tri.csv",
                std::string(base) + "ico_r14_tri.csv",
                std::string(base) + "ico_r20_tri.csv",
                std::string(base) + "ico_r29_tri.csv"
            };
        }

        for (const auto& p : inputs) {
            std::cerr << "[Info] Processing: " << p << "\n";
            process_one_csv(p);
        }

        std::cerr << "[Done] All files processed.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << "\n";
        return 1;
    }
}
