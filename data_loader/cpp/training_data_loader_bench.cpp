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
    const int                  batch_size = 16384;
    const bool                 cyclic     = true;
    const DataloaderSkipConfig config     = {.filtered             = true,
                                             .random_fen_skipping  = 3,
                                             .wld_filtered         = true,
                                             .early_fen_skipping   = 5,
                                             .simple_eval_skipping = 0,
                                             .param_index          = 0,
                                             .pc_y1                = 1.0,
                                             .pc_y2                = 2.0,
                                             .pc_y3                = 1.0};
    const DataloaderDDPConfig  ddp_config = {.rank = 0, .world_size = 1};
    std::unique_ptr<SparseBatchStream, decltype(&destroy_sparse_batch_stream)> stream(
        create_sparse_batch_stream("Full_Threats", concurrency, file_count, files, batch_size, cyclic, config, ddp_config),
        &destroy_sparse_batch_stream);

    auto t0 = std::chrono::high_resolution_clock::now();

#ifdef PGO_BUILD
    constexpr int iteration_count = 100;
#else
    constexpr int iteration_count = 6000;
#endif

    for (int i = 1; i <= iteration_count; ++i)
    {
        {
            std::unique_ptr<SparseBatch, decltype(&destroy_sparse_batch)> b(
                fetch_next_sparse_batch(stream.get()), &destroy_sparse_batch);
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
                      << "   It/s: " << std::setw(8) << std::setprecision(1) << its     //
                      << "   MB/s: " << std::setw(8) << std::setprecision(1) << mbps    //
                      << "   B/pos: " << std::setw(8) << std::setprecision(1) << bpos   //
                      << std::flush;
        }
    }
    std::cout << std::endl;

    return 0;
}
