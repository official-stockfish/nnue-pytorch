#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"

#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif

using namespace binpack;
using namespace chess;

namespace halfkp {

    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static Square orient(Color color, Square sq)
    {
        if (color == Color::White)
        {
            return sq;
        }
        else
        {
            return sq.flippedVertically();
        }
    }

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    struct DenseEntry
    {
        DenseEntry(const TrainingDataEntry& e)
        {
            us = static_cast<float>(e.pos.sideToMove());
            outcome = (e.result + 1.0f) / 2.0f;
            score = e.score;
            std::memset(white, 0, INPUTS * sizeof(float));
            std::memset(black, 0, INPUTS * sizeof(float));
            fill_features(e);
        }

        float us;
        float outcome;
        float score;
        float white[INPUTS];
        float black[INPUTS];

    private:

        void fill_features(const TrainingDataEntry& e, float* features, Color color)
        {
            auto& pos = e.pos;
            auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
            auto ksq = pos.kingSquare(color);
            for(Square sq : pieces)
            {
                auto p = pos.pieceAt(sq);
                features[feature_index(color, orient(color, ksq), sq, p)] = 1.0f;
            }
        }

        void fill_features(const TrainingDataEntry& e)
        {
            fill_features(e, white, Color::White);
            fill_features(e, black, Color::Black);
        }
    };

    struct DenseBatch
    {
        static constexpr int NUM_SQ = 64;
        static constexpr int NUM_PT = 10;
        static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
        static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

        DenseBatch(const std::vector<TrainingDataEntry>& entries)
        {
            size = entries.size();
            us = new float[size];
            outcome = new float[size];
            score = new float[size];
            white = new float[size * INPUTS];
            black = new float[size * INPUTS];

            std::memset(white, 0, size * INPUTS * sizeof(float));
            std::memset(black, 0, size * INPUTS * sizeof(float));

            for(int i = 0; i < entries.size(); ++i)
            {
                fill_entry(i, entries[i]);
            }
        }

        int size;

        float* us;
        float* outcome;
        float* score;
        float* white;
        float* black;

        ~DenseBatch()
        {
            delete[] us;
            delete[] outcome;
            delete[] score;
            delete[] white;
            delete[] black;
        }

    private:

        void fill_entry(int i, const TrainingDataEntry& e)
        {
            us[i] = static_cast<float>(e.pos.sideToMove());
            outcome[i] = (e.result + 1.0f) / 2.0f;
            score[i] = e.score;
            fill_features(i, e);
        }

        void fill_features(const TrainingDataEntry& e, float* features, Color color)
        {
            auto& pos = e.pos;
            auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
            auto ksq = pos.kingSquare(color);
            for(Square sq : pieces)
            {
                auto p = pos.pieceAt(sq);
                features[feature_index(color, orient(color, ksq), sq, p)] = 1.0f;
            }
        }

        void fill_features(int i, const TrainingDataEntry& e)
        {
            fill_features(e, white + i * INPUTS, Color::White);
            fill_features(e, black + i * INPUTS, Color::Black);
        }
    };

    struct SparseEntry
    {
        static constexpr int NUM_SQ = 64;
        static constexpr int NUM_PT = 10;
        static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
        static constexpr int INPUTS = NUM_PLANES * NUM_SQ;
        static constexpr int MAX_ACTIVE_FEATURES = 32;

        SparseEntry(const TrainingDataEntry& e)
        {
            us = static_cast<float>(e.pos.sideToMove());
            outcome = (e.result + 1.0f) / 2.0f;
            score = e.score;
            num_active_white_features = 0;
            num_active_black_features = 0;
            std::fill(std::begin(white), std::end(white), 0.0f);
            std::fill(std::begin(black), std::end(black), 0.0f);
            fill_features(e);
        }

        float us;
        float outcome;
        float score;
        int num_active_white_features;
        int num_active_black_features;
        int white[MAX_ACTIVE_FEATURES];
        int black[MAX_ACTIVE_FEATURES];

    private:

        void fill_features(const TrainingDataEntry& e, int* features, int& counter, Color color)
        {
            auto& pos = e.pos;
            auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
            auto ksq = pos.kingSquare(color);
            for(Square sq : pieces)
            {
                auto p = pos.pieceAt(sq);
                features[counter++] = feature_index(color, orient(color, ksq), sq, p);
            }
        }

        void fill_features(const TrainingDataEntry& e)
        {
            fill_features(e, white, num_active_white_features, Color::White);
            fill_features(e, black, num_active_black_features, Color::Black);
        }
    };

    struct SparseBatch
    {
        static constexpr int NUM_SQ = 64;
        static constexpr int NUM_PT = 10;
        static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
        static constexpr int INPUTS = NUM_PLANES * NUM_SQ;
        static constexpr int MAX_ACTIVE_FEATURES = 32;
        static constexpr int INDEX_DIM = 2;

        SparseBatch(const std::vector<TrainingDataEntry>& entries)
        {
            size = entries.size();
            us = new float[size];
            outcome = new float[size];
            score = new float[size];
            white = new int[size * MAX_ACTIVE_FEATURES * INDEX_DIM];
            black = new int[size * MAX_ACTIVE_FEATURES * INDEX_DIM];

            num_active_white_features = 0;
            num_active_black_features = 0;

            std::memset(white, 0, size * MAX_ACTIVE_FEATURES * INDEX_DIM * sizeof(int));
            std::memset(black, 0, size * MAX_ACTIVE_FEATURES * INDEX_DIM * sizeof(int));

            for(int i = 0; i < entries.size(); ++i)
            {
                fill_entry(i, entries[i]);
            }
        }

        int size;

        float* us;
        float* outcome;
        float* score;
        int num_active_white_features;
        int num_active_black_features;
        int* white;
        int* black;

        ~SparseBatch()
        {
            delete[] us;
            delete[] outcome;
            delete[] score;
            delete[] white;
            delete[] black;
        }

    private:

        void fill_entry(int i, const TrainingDataEntry& e)
        {
            us[i] = static_cast<float>(e.pos.sideToMove());
            outcome[i] = (e.result + 1.0f) / 2.0f;
            score[i] = e.score;
            fill_features(i, e);
        }

        void fill_features(int i, const TrainingDataEntry& e, int* features, int& counter, Color color)
        {
            auto& pos = e.pos;
            auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
            auto ksq = pos.kingSquare(color);
            for(Square sq : pieces)
            {
                auto p = pos.pieceAt(sq);
                int idx = counter * 2;
                counter += 1;
                features[idx] = i;
                features[idx + 1] = feature_index(color, orient(color, ksq), sq, p);
            }
        }

        void fill_features(int i, const TrainingDataEntry& e)
        {
            fill_features(i, e, white, num_active_white_features, Color::White);
            fill_features(i, e, black, num_active_black_features, Color::Black);
        }
    };

    struct DenseEntryStream
    {
        std::unique_ptr<training_data::BasicSfenInputStream> stream;
    };

    struct SparseEntryStream
    {
        std::unique_ptr<training_data::BasicSfenInputStream> stream;
    };

    struct DenseBatchStream
    {
        std::unique_ptr<training_data::BasicSfenInputStream> stream;
    };

    struct SparseBatchStream
    {
        std::unique_ptr<training_data::BasicSfenInputStream> stream;
        std::future<SparseBatch*> next_batch;
    };
}

extern "C" {

    EXPORT halfkp::DenseEntryStream* CDECL create_halfkp_dense_entry_stream(const char* filename, bool cyclic)
    {
        auto stream = training_data::open_sfen_input_file(filename, cyclic);
        return new halfkp::DenseEntryStream{std::move(stream)};
    }

    EXPORT void CDECL destroy_halfkp_dense_entry_stream(halfkp::DenseEntryStream* stream_handle)
    {
        delete stream_handle;
    }

    EXPORT halfkp::SparseEntryStream* CDECL create_halfkp_sparse_entry_stream(const char* filename, bool cyclic)
    {
        auto stream = training_data::open_sfen_input_file(filename, cyclic);
        return new halfkp::SparseEntryStream{std::move(stream)};
    }

    EXPORT void CDECL destroy_halfkp_sparse_entry_stream(halfkp::SparseEntryStream* stream_handle)
    {
        delete stream_handle;
    }

    EXPORT halfkp::DenseBatchStream* CDECL create_halfkp_dense_batch_stream(const char* filename, bool cyclic)
    {
        auto stream = training_data::open_sfen_input_file(filename, cyclic);
        return new halfkp::DenseBatchStream{std::move(stream)};
    }

    EXPORT void CDECL destroy_halfkp_dense_batch_stream(halfkp::DenseBatchStream* stream_handle)
    {
        delete stream_handle;
    }

    EXPORT halfkp::SparseBatchStream* CDECL create_halfkp_sparse_batch_stream(const char* filename, bool cyclic)
    {
        auto stream = training_data::open_sfen_input_file(filename, cyclic);
        return new halfkp::SparseBatchStream{std::move(stream)};
    }

    EXPORT void CDECL destroy_halfkp_sparse_batch_stream(halfkp::SparseBatchStream* stream_handle)
    {
        if(stream_handle->next_batch.valid())
        {
            delete stream_handle->next_batch.get();
        }

        delete stream_handle;
    }

    EXPORT halfkp::DenseEntry* CDECL get_next_halfkp_dense_entry(halfkp::DenseEntryStream* stream_handle)
    {
        auto& stream = *(stream_handle->stream);
        auto value = stream.next();
        if (value.has_value())
        {
            return new halfkp::DenseEntry(*value);
        }
        else
        {
            return nullptr;
        }
    }

    EXPORT void CDECL destroy_halfkp_dense_entry(halfkp::DenseEntry* e)
    {
        delete e;
    }

    EXPORT halfkp::SparseEntry* CDECL get_next_halfkp_sparse_entry(halfkp::SparseEntryStream* stream_handle)
    {
        auto& stream = *(stream_handle->stream);
        auto value = stream.next();
        if (value.has_value())
        {
            return new halfkp::SparseEntry(*value);
        }
        else
        {
            return nullptr;
        }
    }

    EXPORT void CDECL destroy_halfkp_sparse_entry(halfkp::SparseEntry* e)
    {
        delete e;
    }

    EXPORT halfkp::DenseBatch* CDECL get_next_halfkp_dense_batch(halfkp::DenseBatchStream* stream_handle, int max_batch_size)
    {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(max_batch_size);
        auto& stream = *(stream_handle->stream);

        for(int i = 0; i < max_batch_size; ++i)
        {
            auto value = stream.next();
            if (value.has_value())
            {
                entries.emplace_back(*value);
            }
            else
            {
                break;
            }
        }

        return new halfkp::DenseBatch(entries);
    }

    EXPORT void CDECL destroy_halfkp_dense_batch(halfkp::DenseBatch* e)
    {
        delete e;
    }

    EXPORT halfkp::SparseBatch* CDECL get_next_halfkp_sparse_batch(halfkp::SparseBatchStream* stream_handle, int max_batch_size)
    {
        for(;;)
        {
            auto cur = std::move(stream_handle->next_batch);
            if (cur.valid())
            {
                // we have to wait for this to complete before scheduling the next one
                cur.wait();
            }

            stream_handle->next_batch = std::async(std::launch::async, [stream_handle, max_batch_size]() {
                std::vector<TrainingDataEntry> entries;
                entries.reserve(max_batch_size);
                auto& stream = *(stream_handle->stream);

                for(int i = 0; i < max_batch_size; ++i)
                {
                    auto value = stream.next();
                    if (value.has_value())
                    {
                        entries.emplace_back(*value);
                    }
                    else
                    {
                        break;
                    }
                }

                return new halfkp::SparseBatch(entries);
            });

            if (cur.valid())
            {
                return cur.get();
            }
        }
    }

    EXPORT void CDECL destroy_halfkp_sparse_batch(halfkp::SparseBatch* e)
    {
        delete e;
    }

}
