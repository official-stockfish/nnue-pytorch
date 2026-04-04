#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <thread>
#include <filesystem>
#include <vector>
#include <memory>
#include <stdexcept>
#include <map>
#include <sstream>
#include <algorithm>

#include "training_data_loader_abi.h"
#include "training_data_loader_internal.h"

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// DELETERS
// -----------------------------------------------------------------------------

struct SparseBatchDeleter {
    void operator()(SparseBatch* b) const {
        destroy_sparse_batch(b);
    }
};

struct SparseBatchStreamDeleter {
    void operator()(SparseBatchStream* s) const {
        destroy_sparse_batch_stream(s);
    }
};

// -----------------------------------------------------------------------------
// CONFIGURATION STRUCTS & PARSING
// -----------------------------------------------------------------------------

struct CliConfig {
    DataloaderSkipConfig skip_config;
    DataloaderDDPConfig  ddp_config;
    int                  batch_size;
    bool                 cyclic;
};

const CliConfig default_cli_config = {
    .skip_config = {
        .filtered             = true,
        .random_fen_skipping  = 10,
        .wld_filtered         = true,
        .early_fen_skipping   = 10,
        .simple_eval_skipping = 0,
        .param_index          = 0,
        .pc_y1                = 0.6893201149773951,
        .pc_y2                = 2.9285769485515805,
        .pc_y3                = 1.4386005301749225
    },
    .ddp_config = {.rank = 0, .world_size = 1},
    .batch_size = 65536,
    .cyclic     = true
};

// Robustly get the directory containing the current executable (Linux specific)
fs::path get_executable_dir() {
    std::error_code ec;
    fs::path exe_path = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        return exe_path.parent_path();
    }
    return fs::current_path(); // Fallback if /proc/self/exe is somehow unavailable
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Parses a simple key=value INI file. Ignores comments (#) and sections ([]).
bool parse_ini(const fs::path& filepath, std::map<std::string, std::string>& config_map) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        if (line[0] == '[') continue; // Ignore section headers

        size_t delim = line.find('=');
        if (delim != std::string::npos) {
            std::string key = trim(line.substr(0, delim));
            std::string val = trim(line.substr(delim + 1));
            config_map[key] = val;
        }
    }
    return true;
}

CliConfig build_config_from_map(const std::map<std::string, std::string>& m) {
    const std::vector<std::string> required_keys = {
        "filtered", "random_fen_skipping", "wld_filtered", "early_fen_skipping",
        "simple_eval_skipping", "param_index", "pc_y1", "pc_y2", "pc_y3",
        "ddp_config.rank", "ddp_config.world_size", "batch_size", "cyclic"
    };

    for (const auto& key : required_keys) {
        if (m.find(key) == m.end()) {
            throw std::runtime_error("Missing required key: " + key);
        }
    }

    auto parse_bool = [](const std::string& s) {
        std::string lower_s = s;
        std::transform(lower_s.begin(), lower_s.end(), lower_s.begin(), ::tolower);
        return (lower_s == "1" || lower_s == "true");
    };

    return {
        .skip_config = {
            .filtered             = parse_bool(m.at("filtered")),
            .random_fen_skipping  = std::stoi(m.at("random_fen_skipping")),
            .wld_filtered         = parse_bool(m.at("wld_filtered")),
            .early_fen_skipping   = std::stoi(m.at("early_fen_skipping")),
            .simple_eval_skipping = std::stoi(m.at("simple_eval_skipping")),
            .param_index          = std::stoi(m.at("param_index")),
            .pc_y1                = std::stod(m.at("pc_y1")),
            .pc_y2                = std::stod(m.at("pc_y2")),
            .pc_y3                = std::stod(m.at("pc_y3"))
        },
        .ddp_config = {
            .rank       = std::stoi(m.at("ddp_config.rank")),
            .world_size = std::stoi(m.at("ddp_config.world_size"))
        },
        .batch_size = std::stoi(m.at("batch_size")),
        .cyclic     = parse_bool(m.at("cyclic"))
    };
}

#ifdef NNUE_LOADER_STATISTICS

// -----------------------------------------------------------------------------
// REPORT IMPLEMENTATION
// -----------------------------------------------------------------------------

struct DistributionReport {
    uint64_t pc_counts[33] = {0};
    uint64_t total_count = 0;
    std::vector<uint64_t> ply_counts;
    size_t   max_plies_plus_one;

    DistributionReport(size_t max_plies)
        : ply_counts(max_plies + 2),
          max_plies_plus_one(max_plies + 1)
    {}

    void add_batch(const SparseBatch* batch, int b_size) {
        if (!batch) return;

        for (int i = 0; i < b_size; ++i) {
            int pc = batch->entries_copy[i].pos.piecesBB().count();
            int ply = batch->entries_copy[i].ply;

            if (pc >= 0 && pc <= 32) {
                pc_counts[pc]++;
            }

            if (ply >= 0) {
                int ply_index = (ply > max_plies_plus_one) ? max_plies_plus_one : ply;
                ply_counts[ply_index]++;
            }

            total_count++;
        }
    }

    void print_histogram(const std::string& title, const uint64_t* counts, int size, bool is_ply) const {
        std::cout << "\n=== " << title << " ===" << std::endl;
        std::cout << std::setw(5) << (is_ply ? "Ply" : "PC") << " | "
                  << std::setw(12) << "Count" << " | Share" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        for (int i = 0; i < size; ++i) {
            if (counts[i] == 0) continue;
            double share = (total_count > 0) ? (double)counts[i] / total_count : 0;
            int bar_width = static_cast<int>(share * 60);

            std::string label = (is_ply && i == size - 1) ? ">=" + std::to_string(i) : std::to_string(i);

            std::cout << std::setw(5) << label << " | "
                      << std::setw(12) << counts[i] << " | "
                      << std::fixed << std::setprecision(2) << std::setw(6) << (share * 100.0) << "% "
                      << std::string(bar_width, '#') << std::endl;
        }
        std::cout << "--------------------------------------------------" << std::endl;
    }

    void print() const {
        if (total_count == 0) {
            std::cout << "\nNo data processed." << std::endl;
            return;
        }
        print_histogram("Piece Count Distribution", pc_counts, 33, false);
        print_histogram("Ply Distribution", ply_counts.data(), max_plies_plus_one + 1, true);
        std::cout << "Total entries tracked: " << total_count << std::endl;
    }
};

void run_report(int concurrency, size_t iteration_count, size_t max_plies, int file_count, const char** files, CliConfig cli_config) {
    auto skip_config = cli_config.skip_config;
    auto ddp_config = cli_config.ddp_config;
    int batch_size = cli_config.batch_size;
    bool cyclic = cli_config.cyclic;

    std::cout << "Initializing stream (Threads: " << concurrency << ", Iterations: " << iteration_count << ")..." << std::endl;

    std::unique_ptr<SparseBatchStream, SparseBatchStreamDeleter> stream(
        create_sparse_batch_stream("Full_Threats+HalfKAv2_hm", concurrency, file_count, files,
            batch_size, cyclic, skip_config, ddp_config));

    DistributionReport report(max_plies);

    std::cout << "Sampling dataloader stream..." << std::endl;

    for (size_t iter = 1; iter <= iteration_count; ++iter) {
        std::unique_ptr<SparseBatch, SparseBatchDeleter> b(fetch_next_sparse_batch(stream.get()));

        if (b) {
            report.add_batch(b.get(), batch_size);
        }

        if (iter % 100 == 0) {
            std::cout << "\rBatches sampled: " << iter << " / " << iteration_count << std::flush;
        }
    }

    std::cout << "\nFinished sampling." << std::endl;
    report.print();
}

#else

// -----------------------------------------------------------------------------
// BENCH IMPLEMENTATION
// -----------------------------------------------------------------------------

std::vector<std::string> stage_files_to_ram(int file_count, const char** files) {
    std::vector<std::string> ram_paths;
    ram_paths.reserve(file_count);

    const fs::path ram_base = "/dev/shm/app_cache";

    try {
        if (!fs::exists(ram_base)) {
            fs::create_directories(ram_base);
        }

        for (int i = 0; i < file_count; ++i) {
            if (files[i] == nullptr) continue;

            fs::path original_path(files[i]);

            if (!fs::exists(original_path) || !fs::is_regular_file(original_path)) {
                throw std::runtime_error("Invalid or missing file: " + original_path.string());
            }

            fs::path target_path = ram_base / original_path.filename();

            if (!fs::exists(target_path)) {
                fs::copy_file(original_path, target_path, fs::copy_options::overwrite_existing);
            }

            ram_paths.push_back(target_path.string());
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        throw;
    }

    return ram_paths;
}

long long get_rchar_self() {
    std::ifstream io_file("/proc/self/io");
    std::string   line;
    while (std::getline(io_file, line)) {
        if (line.rfind("rchar:", 0) == 0) {
            return std::stoll(line.substr(6));
        }
    }
    return -1;
}

void run_bench(int concurrency, size_t iteration_count, int do_cache_files, int file_count, const char** files, CliConfig cli_config) {
    auto skip_config = cli_config.skip_config;
    auto ddp_config = cli_config.ddp_config;
    int batch_size = cli_config.batch_size;
    bool cyclic = cli_config.cyclic;

    std::vector<std::string> ram_files;
    std::vector<const char*> c_str_paths;

    std::cout << "Threads: " << concurrency << " | Iterations: " << iteration_count << "\n";

    if (do_cache_files == 1) {
        std::cout << "Caching files to ram: ..." << std::endl;
        ram_files = stage_files_to_ram(file_count, files);
        for (const auto& path : ram_files) {
            c_str_paths.push_back(path.c_str());
        }
        file_count = static_cast<int>(c_str_paths.size());
        files = c_str_paths.data();
        std::cout << "Caching files to ram: done" << std::endl;
    }

    std::unique_ptr<SparseBatchStream, SparseBatchStreamDeleter> stream(
        create_sparse_batch_stream("Full_Threats+HalfKAv2_hm", concurrency, file_count, files,
            batch_size, cyclic, skip_config, ddp_config));

    size_t warmup_iterations = 5;
    for (size_t i = 1; i <= warmup_iterations; ++i) {
        std::unique_ptr<SparseBatch, SparseBatchDeleter> b(fetch_next_sparse_batch(stream.get()));
    }

    long long bytes_before = get_rchar_self();
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t i = 1; i <= iteration_count; ++i) {
        std::unique_ptr<SparseBatch, SparseBatchDeleter> b(fetch_next_sparse_batch(stream.get()));

        auto t1 = std::chrono::high_resolution_clock::now();
        // i % 1 == 0 check from original preserved explicitly
        if (i % 1 == 0) {
            double    sec   = std::chrono::duration<double>(t1 - t0).count();
            long long bytes = get_rchar_self() - bytes_before;

            double mpos = i * batch_size / (sec * 1000 * 1000);
            double its  = i / sec;
            double mbps = bytes / (sec * 1024 * 1024);
            double bpos = bytes / (i * batch_size);

            std::cout << "\rIter: " << std::setw(8) << i
                      << "   Time(s): " << std::setw(8) << std::setprecision(3) << sec
                      << "   MPos/s: " << std::setw(8) << std::setprecision(3) << mpos
                      << "   It/s: " << std::setw(8) << std::setprecision(3) << its
                      << "   MB/s: " << std::setw(8) << std::setprecision(3) << mbps
                      << "   B/pos: " << std::setw(8) << std::setprecision(3) << bpos
                      << std::flush;
        }
    }
    std::cout << std::endl;
}

#endif

// -----------------------------------------------------------------------------
// UNIFIED ENTRY POINT
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    int concurrency = std::thread::hardware_concurrency();
    size_t iteration_count = 1000;
    size_t max_plies = 100;
    int do_cache_files = 1;
    std::string cli_settings_path = "";

    int i = 1;
    for (; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-p" && i + 1 < argc) {
            concurrency = std::stoi(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            iteration_count = std::stoul(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            do_cache_files = std::stoi(argv[++i]);
        } else if (arg == "-m" && i + 1 < argc) {
            max_plies = std::stoul(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            cli_settings_path = argv[++i];
        } else if (arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        } else {
            break;
        }
    }

    if (i >= argc) {
        std::cerr << "Usage: " << argv[0] << " [-i iterations] [-p concurrency] [-c do_cache_files] [-m max_plies] [-s config.ini] file1 [file2 ...]\n";
        return 1;
    }

// --- Configuration Resolution ---
    CliConfig active_config = default_cli_config;
    fs::path target_config_path;
    bool using_custom_config = false;

    if (!cli_settings_path.empty()) {
        target_config_path = fs::absolute(cli_settings_path);
        if (!fs::exists(target_config_path) || !fs::is_regular_file(target_config_path)) {
            std::cerr << "FATAL: Explicitly requested config file not found at " << target_config_path << "\n";
            std::exit(1);
        }
    } else {
        target_config_path = get_executable_dir() / "data_loader_config.ini";
    }

    if (fs::exists(target_config_path) && fs::is_regular_file(target_config_path)) {
        std::map<std::string, std::string> parsed_ini;
        if (!parse_ini(target_config_path, parsed_ini)) {
            std::cerr << "FATAL: Failed to read config file at " << target_config_path << "\n";
            std::exit(1);
        }
        try {
            active_config = build_config_from_map(parsed_ini);
            using_custom_config = true;
            std::cout << "Configuration loaded successfully from: " << target_config_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "FATAL: Config file at " << target_config_path
                      << " is incomplete or invalid (" << e.what() << "). Aborting.\n";
            std::exit(1);
        }
    } else if (cli_settings_path.empty()) {
        // Only allow fallback if no CLI flag was provided AND default file doesn't exist
        std::cout << "No local config found at " << target_config_path << ". Using hardcoded defaults.\n";
    }
    // --------------------------------

    const char** files = const_cast<const char**>(&argv[i]);
    int file_count = argc - i;

    if (concurrency < 1) concurrency = 1;
    if (iteration_count < 1) iteration_count = 1;

#ifdef NNUE_LOADER_STATISTICS
    run_report(concurrency, iteration_count, max_plies, file_count, files, active_config);
#else
    run_bench(concurrency, iteration_count, do_cache_files, file_count, files, active_config);
#endif

    return 0;
}
