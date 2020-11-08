#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>

#include "nnue_data_binpack_format.h"
#include "training_data_stream.h"

#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif

using namespace binpack;
using namespace chess;

struct TestDataCollection
{
    int size;
    int* data;
};

struct TrainingEntryHalfKPDense
{
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    TrainingEntryHalfKPDense(const TrainingDataEntry& e)
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

    static int halfkp_idx(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    void fill_features(const TrainingDataEntry& e, float* features, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
        auto ksq = pos.kingSquare(color);
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            features[halfkp_idx(color, orient(color, ksq), sq, p)] = 1.0f;
        }
    }

    void fill_features(const TrainingDataEntry& e)
    {
        fill_features(e, white, Color::White);
        fill_features(e, black, Color::Black);
    }
};

struct TrainingEntryHalfKPDenseBatch
{
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    TrainingEntryHalfKPDenseBatch(const std::vector<TrainingDataEntry>& entries)
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

    ~TrainingEntryHalfKPDenseBatch()
    {
        delete[] us;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
    }

private:

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

    static int halfkp_idx(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

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
            features[halfkp_idx(color, orient(color, ksq), sq, p)] = 1.0f;
        }
    }

    void fill_features(int i, const TrainingDataEntry& e)
    {
        fill_features(e, white + i * INPUTS, Color::White);
        fill_features(e, black + i * INPUTS, Color::Black);
    }
};

struct TrainingEntryHalfKPSparse
{
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;
    static constexpr int MAX_ACTIVE_FEATURES = 32;

    TrainingEntryHalfKPSparse(const TrainingDataEntry& e)
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

    static int halfkp_idx(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    void fill_features(const TrainingDataEntry& e, int* features, int& counter, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
        auto ksq = pos.kingSquare(color);
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            features[counter++] = halfkp_idx(color, orient(color, ksq), sq, p);
        }
    }

    void fill_features(const TrainingDataEntry& e)
    {
        fill_features(e, white, num_active_white_features, Color::White);
        fill_features(e, black, num_active_black_features,Color::Black);
    }
};

struct TrainingEntryHalfKPSparseBatch
{
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;
    static constexpr int MAX_ACTIVE_FEATURES = 32;
    static constexpr int INDEX_DIM = 2;

    TrainingEntryHalfKPSparseBatch(const std::vector<TrainingDataEntry>& entries)
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

    ~TrainingEntryHalfKPSparseBatch()
    {
        delete[] us;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
    }

private:

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

    static int halfkp_idx(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

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
            features[idx + 1] = halfkp_idx(color, orient(color, ksq), sq, p);
        }
    }

    void fill_features(int i, const TrainingDataEntry& e)
    {
        fill_features(i, e, white, num_active_white_features, Color::White);
        fill_features(i, e, black, num_active_black_features,Color::Black);
    }
};

struct InputStreamHandle
{
    std::unique_ptr<training_data::BasicSfenInputStream> stream;
    std::future<TrainingEntryHalfKPSparseBatch*> next_sparse_batch;
};

extern "C" {

    EXPORT void CDECL test()
    {
        std::cout<< "test successful\n";
    }

    EXPORT TestDataCollection* CDECL create_data_collection()
    {
        return new TestDataCollection{ 10, new int[10]{} };
    }

    EXPORT void CDECL destroy_data_collection(TestDataCollection* ptr)
    {
        delete ptr->data;
        delete ptr;
    }

    EXPORT InputStreamHandle* CDECL create_stream(const char* filename, bool cyclic)
    {
        auto stream = training_data::open_sfen_input_file(filename, cyclic);
        return new InputStreamHandle{std::move(stream)};
    }

    EXPORT void CDECL destroy_stream(InputStreamHandle* stream_handle)
    {
        if(stream_handle->next_sparse_batch.valid())
        {
            delete stream_handle->next_sparse_batch.get();
        }

        delete stream_handle;
    }

    EXPORT void CDECL print_next(InputStreamHandle* stream_handle)
    {
        auto& stream = *(stream_handle->stream);
        auto value = stream.next();
        if (value.has_value())
        {
            std::cout << value->pos.fen() << '\n';
        }
        else
        {
            std::cout << "Nothing.\n";
        }
    }

    EXPORT TrainingEntryHalfKPDense* CDECL get_next_entry_halfkp_dense(InputStreamHandle* stream_handle)
    {
        auto& stream = *(stream_handle->stream);
        auto value = stream.next();
        if (value.has_value())
        {
            return new TrainingEntryHalfKPDense(*value);
        }
        else
        {
            return nullptr;
        }
    }

    EXPORT TrainingEntryHalfKPSparse* CDECL get_next_entry_halfkp_sparse(InputStreamHandle* stream_handle)
    {
        auto& stream = *(stream_handle->stream);
        auto value = stream.next();
        if (value.has_value())
        {
            return new TrainingEntryHalfKPSparse(*value);
        }
        else
        {
            return nullptr;
        }
    }

    EXPORT void CDECL destroy_entry_halfkp_dense(TrainingEntryHalfKPDense* e)
    {
        delete e;
    }

    EXPORT void CDECL destroy_entry_halfkp_sparse(TrainingEntryHalfKPSparse* e)
    {
        delete e;
    }

    EXPORT TrainingEntryHalfKPDenseBatch* CDECL get_next_entry_halfkp_dense_batch(InputStreamHandle* stream_handle, int max_batch_size)
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

        return new TrainingEntryHalfKPDenseBatch(entries);
    }

    EXPORT void CDECL destroy_entry_halfkp_dense_batch(TrainingEntryHalfKPDenseBatch* e)
    {
        delete e;
    }

    EXPORT TrainingEntryHalfKPSparseBatch* CDECL get_next_entry_halfkp_sparse_batch(InputStreamHandle* stream_handle, int max_batch_size)
    {
        for(;;)
        {
            auto cur = std::move(stream_handle->next_sparse_batch);
            if (cur.valid())
            {
                // we have to wait for this to complete before scheduling the next one
                cur.wait();
            }

            stream_handle->next_sparse_batch = std::async(std::launch::async, [stream_handle, max_batch_size]() {
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

                return new TrainingEntryHalfKPSparseBatch(entries);
            });

            if (cur.valid())
            {
                return cur.get();
            }
        }
    }

    EXPORT void CDECL destroy_entry_halfkp_sparse_batch(TrainingEntryHalfKPSparseBatch* e)
    {
        delete e;
    }

}
