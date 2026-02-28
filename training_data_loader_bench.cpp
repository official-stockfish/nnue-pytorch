/*

// Option 1: build by compiling the implementation directly into the binary
// (uses training_data_loader.cpp)
g++ -std=c++20 -g3 -O3 -DNDEBUG -DBENCH -march=native \
    training_data_loader_bench.cpp \
    training_data_loader.cpp \
    -o bench
// Option 2: build by linking against the shared library (recommended to
// match the README examples and typical usage)
g++ -std=c++20 -g3 -O3 -DNDEBUG -DBENCH -march=native \
    training_data_loader_bench.cpp \
    -L. -ltraining_data_loader -Wl,-rpath,'$ORIGIN' \
    -o bench

./bench /path/to/binpack
*/

#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <thread>

#include "training_data_loader_abi.h"

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

int main(int argc, char** argv) {
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " file1 [file2 ...]\n";
        return 1;
    }
    const char** files      = const_cast<const char**>(&argv[1]);
    int          file_count = argc - 1;

#ifdef PGO_BUILD
    const int concurrency = 1;
#else
    const int concurrency = std::thread::hardware_concurrency();
#endif
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
    auto stream = create_sparse_batch_stream("Full_Threats", concurrency, file_count, files,
                                             batch_size, cyclic, config, ddp_config);

    auto t0 = std::chrono::high_resolution_clock::now();

#ifdef PGO_BUILD
    constexpr size_t iteration_count = 10;
#else
    constexpr size_t iteration_count = 6000;
#endif

    for (size_t i = 1; i <= iteration_count; ++i)
    {
        if (auto* b = fetch_next_sparse_batch(stream))
        {
            destroy_sparse_batch(b);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        if (i % 1 == 0)
        {
            double    sec   = (t1 - t0).count() / 1e9;
            long long bytes = get_rchar_self();

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
