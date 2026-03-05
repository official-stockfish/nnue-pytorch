/*

// Option 1: build by compiling the implementation directly into the binary
// (uses training_data_loader.cpp)
g++ -std=c++20 -g3 -O3 -DNDEBUG -DBENCH -march=native \
    data_loader/cpp/training_data_loader_bench.cpp \
    data_loader/cpp/training_data_loader.cpp \
    -o bench_static
// Option 2: build by linking against the shared library (recommended to
// match the README examples and typical usage)
// if you haven't built the shared library yet, do so first with:
g++ -std=c++20 -g3 -O3 -DNDEBUG -march=native -fPIC -shared \
    data_loader/cpp/training_data_loader.cpp \
    -o build/libtraining_data_loader.so
// then build the benchmark linking against the shared library:
g++ -std=c++20 -g3 -O3 -DNDEBUG -DBENCH -march=native \
    data_loader/cpp/training_data_loader_bench.cpp \
    -L./build -ltraining_data_loader -Wl,-rpath,'$ORIGIN/build' \
    -o bench_shared

./bench /path/to/binpack
*/

#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <thread>
#include <filesystem>
#include <vector>

#include "training_data_loader_abi.h"

namespace fs = std::filesystem;

/**
 * Validates existence, copies files to /dev/shm for low-latency access,
 * and returns the new RAM-based paths.
 */
std::vector<std::string> stage_files_to_ram(int file_count, const char** files) {
    std::vector<std::string> ram_paths;
    ram_paths.reserve(file_count);

    // Using /dev/shm for Linux RAM-disk.
    // For cross-platform, you would need a custom memory buffer interface.
    const fs::path ram_base = "/dev/shm/app_cache";

    try {
        if (!fs::exists(ram_base)) {
            fs::create_directories(ram_base);
        }

        for (int i = 0; i < file_count; ++i) {
            if (files[i] == nullptr) continue;

            fs::path original_path(files[i]);

            // 1. Logic/Existence Validation
            if (!fs::exists(original_path) || !fs::is_regular_file(original_path)) {
                throw std::runtime_error("Invalid or missing file: " + original_path.string());
            }

            // 2. Performance: Copy to RAM
            fs::path target_path = ram_base / original_path.filename();

            // Avoid redundant copies if multiple streams use the same files
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
    while (std::getline(io_file, line))
    {
        if (line.rfind("rchar:", 0) == 0)
        {
            return std::stoll(line.substr(6));
        }
    }
    return -1;  // Error or not found
}

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

int main(int argc, char** argv) {
    int concurrency = std::thread::hardware_concurrency();
    size_t iteration_count = 6000;
    size_t warmup_iterations = 5;
    int do_cache_files = 1;

    int i = 1;
    for (; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-p" && i + 1 < argc) {
            concurrency = std::stoi(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            iteration_count = std::stoul(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            do_cache_files = std::stoi(argv[++i]);
        } else if (arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        } else {
            break;
        }
    }

    if (i >= argc) {
        std::cerr << "Usage: " << argv[0] << " [-i iterations] [-p concurrency] [-c do_cache_files] file1 [file2 ...]\n";
        return 1;
    }

    const char** files = const_cast<const char**>(&argv[i]);
    int file_count = argc - i;
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

    if (concurrency < 1) concurrency = 1;
    if (iteration_count < 1) iteration_count = 1;

    // some typical numbers, more skipping means more load
    const int                  batch_size = 65536;
    const bool                 cyclic     = true;
    const DataloaderSkipConfig config     = {.filtered             = true,
                                             .random_fen_skipping  = 10,
                                             .wld_filtered         = true,
                                             .early_fen_skipping   = 28,
                                             .simple_eval_skipping = 0,
                                             .param_index          = 0,
                                             .pc_y1                = 1.0,
                                             .pc_y2                = 2.0,
                                             .pc_y3                = 1.0};
    const DataloaderDDPConfig  ddp_config = {.rank = 0, .world_size = 1};

    std::unique_ptr<SparseBatchStream, SparseBatchStreamDeleter> stream(
        create_sparse_batch_stream("Full_Threats+HalfKAv2_hm", concurrency, file_count, files,
            batch_size, cyclic, config, ddp_config));


    for (size_t i = 1; i <= warmup_iterations; ++i)
    {
        {
            std::unique_ptr<SparseBatch, SparseBatchDeleter> b(
                fetch_next_sparse_batch(stream.get()));
        }
    }

    long long bytes_before = get_rchar_self();
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t i = 1; i <= iteration_count; ++i)
    {
        {
            std::unique_ptr<SparseBatch, SparseBatchDeleter> b(
                fetch_next_sparse_batch(stream.get()));
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        if (i % 1 == 0)
        {
            double    sec   = std::chrono::duration<double>(t1 - t0).count();
            long long bytes = get_rchar_self() - bytes_before;

            double mpos = i * batch_size / (sec * 1000 * 1000);
            double its  = i / sec;
            double mbps = bytes / (sec * 1024 * 1024);
            double bpos = bytes / (i * batch_size);

            std::cout << "\rIter: " << std::setw(8) << i                                //
                      << "   Time(s): " << std::setw(8) << std::setprecision(3) << sec  //
                      << "   MPos/s: " << std::setw(8) << std::setprecision(3) << mpos  //
                      << "   It/s: " << std::setw(8) << std::setprecision(3) << its     //
                      << "   MB/s: " << std::setw(8) << std::setprecision(3) << mbps    //
                      << "   B/pos: " << std::setw(8) << std::setprecision(3) << bpos   //
                      << std::flush;
        }
    }
    std::cout << std::endl;

    return 0;
}
