#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>

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

struct InputStreamHandle
{
    std::unique_ptr<training_data::BasicSfenInputStream> stream;
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
        std::fill(std::begin(white), std::end(white), 0.0f);
        std::fill(std::begin(black), std::end(black), 0.0f);
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

    EXPORT InputStreamHandle* CDECL create_stream(const char* filename)
    {
        auto stream = training_data::open_sfen_input_file(filename);
        return new InputStreamHandle{std::move(stream)};
    }

    EXPORT void CDECL destroy_stream(InputStreamHandle* stream_handle)
    {
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

}
