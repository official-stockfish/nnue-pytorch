#include <iostream>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>

#include "training_data_loader_abi.h"
#include "training_data_loader_internal.h"

struct DistributionReport {
    uint64_t pc_counts[33] = {0};
    uint64_t total_count = 0;
    std::vector<uint64_t> ply_counts;
    size_t   max_plies_plus_one;

    DistributionReport(size_t max_plies)
        : ply_counts(max_plies + 2),
          max_plies_plus_one(max_plies + 1)
    {}

    void add_batch(const SparseBatch* batch, int batch_size) {
        if (!batch) return;

        for (int i = 0; i < batch_size; ++i) {
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
    size_t iteration_count = 1000;
    size_t max_plies = 100;

    int i = 1;
    for (; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-p" && i + 1 < argc) {
            concurrency = std::stoi(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            iteration_count = std::stoul(argv[++i]);
        } else if (arg == "-m" && i + 1 < argc) {
            max_plies = std::stoul(argv[++i]);
        } else if (arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        } else {
            break;
        }
    }

    if (i >= argc) {
        std::cerr << "Usage: " << argv[0] << " [-i iterations] [-p concurrency] file1 [file2 ...]\n";
        return 1;
    }

    const char** files = const_cast<const char**>(&argv[i]);
    int file_count = argc - i;

    if (concurrency < 1) concurrency = 1;
    if (iteration_count < 1) iteration_count = 1;

    const int                  batch_size = 65536;
    const bool                 cyclic     = true;
    const DataloaderSkipConfig config     = {.filtered             = true,
                                             .random_fen_skipping  = 10,
                                             .wld_filtered         = true,
                                             .early_fen_skipping   = 10,
                                             .simple_eval_skipping = 0,
                                             .param_index          = 0,
                                             .pc_y1                = 0.6893201149773951,
                                             .pc_y2                = 2.9285769485515805,
                                             .pc_y3                = 1.4386005301749225};
    const DataloaderDDPConfig  ddp_config = {.rank = 0, .world_size = 1};

    std::cout << "Initializing stream (Threads: " << concurrency << ", Iterations: " << iteration_count << ")..." << std::endl;

    std::unique_ptr<SparseBatchStream, SparseBatchStreamDeleter> stream(
        create_sparse_batch_stream("Full_Threats+HalfKAv2_hm", concurrency, file_count, files,
            batch_size, cyclic, config, ddp_config));

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

    return 0;
}
