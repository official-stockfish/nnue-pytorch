/*

Copyright 2020 Tomasz Sobczyk

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software
and associated documentation files (the "Software"),
to deal in the Software without restriction,
including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

#pragma once

#include <algorithm>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <climits>
#include <ctime>


#include "chess.h"
#include "arithmetic.h"


namespace binpack {
    struct TrainingDataEntry
    {
        chess::Position pos;
        chess::Move move;
        std::int16_t score;
        std::uint16_t ply;
        std::int16_t result;

        [[nodiscard]] bool isValid() const
        {
            return pos.isMoveLegal(move);
        }

        [[nodiscard]] bool isCapturingMove() const
        {
            return pos.pieceAt(move.to) != chess::Piece::none() &&
                   pos.pieceAt(move.to).color() != pos.pieceAt(move.from).color(); // Exclude castling
        }

        // The win rate model returns the probability (per mille) of winning given an eval
        // and a game-ply. The model fits rather accurately the LTC fishtest statistics.
        std::tuple<double, double, double> win_rate_model() const {

           // The model captures only up to 240 plies, so limit input (and rescale)
           double m = std::min(240, int(ply)) / 64.0;

           // Coefficients of a 3rd order polynomial fit based on fishtest data
           // for two parameters needed to transform eval to the argument of a
           // logistic function.
           double as[] = {-3.68389304,  30.07065921, -60.52878723, 149.53378557};
           double bs[] = {-2.0181857,   15.85685038, -29.83452023,  47.59078827};
           double a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];
           double b = (((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3];

           // tweak wdl model, deviating from fishtest results,
           // but yielding improved training results
           b *= 1.5;

           // Transform eval to centipawns with limited range
           double x = std::clamp(double(100 * score) / 208, -2000.0, 2000.0);
           double w = 1.0 / (1 + std::exp((a - x) / b));
           double l = 1.0 / (1 + std::exp((a + x) / b));
           double d = 1.0 - w - l;

           // Return win, loss, draw rate in per mille (rounded to nearest)
           return std::make_tuple(w, l, d);
        }

        // how likely is end-game result with the current score?
        double score_result_prob() const {
           auto [w, l, d] = win_rate_model();
           if (result > 0)
               return w;
           if (result < 0)
               return l;
           return d;
        }

        [[nodiscard]] bool isInCheck() const
        {
            return pos.isCheck();
        }
    };


    [[nodiscard]] inline bool isContinuation(const TrainingDataEntry& lhs, const TrainingDataEntry& rhs)
    {
        return
            lhs.result == -rhs.result
            && lhs.ply + 1 == rhs.ply
            && lhs.pos.afterMove(lhs.move) == rhs.pos;
    }

    struct PackedTrainingDataEntry
    {
        unsigned char bytes[32];
    };

    [[nodiscard]] inline PackedTrainingDataEntry packEntry(const TrainingDataEntry& plain)
    {
        PackedTrainingDataEntry packed;

        auto compressedPos = plain.pos.compress();
        auto compressedMove = plain.move.compress();

        static_assert(sizeof(compressedPos) + sizeof(compressedMove) + 6 == sizeof(PackedTrainingDataEntry));

        std::size_t offset = 0;
        compressedPos.writeToBigEndian(packed.bytes);
        offset += sizeof(compressedPos);
        compressedMove.writeToBigEndian(packed.bytes + offset);
        offset += sizeof(compressedMove);
        std::uint16_t pr = plain.ply | (signedToUnsigned(plain.result) << 14);
        packed.bytes[offset++] = signedToUnsigned(plain.score) >> 8;
        packed.bytes[offset++] = signedToUnsigned(plain.score);
        packed.bytes[offset++] = pr >> 8;
        packed.bytes[offset++] = pr;
        packed.bytes[offset++] = plain.pos.rule50Counter() >> 8;
        packed.bytes[offset++] = plain.pos.rule50Counter();

        return packed;
    }

    [[nodiscard]] inline TrainingDataEntry unpackEntry(const PackedTrainingDataEntry& packed)
    {
        TrainingDataEntry plain;

        std::size_t offset = 0;
        auto compressedPos = chess::CompressedPosition::readFromBigEndian(packed.bytes);
        plain.pos = compressedPos.decompress();
        offset += sizeof(compressedPos);
        auto compressedMove = chess::CompressedMove::readFromBigEndian(packed.bytes + offset);
        plain.move = compressedMove.decompress();
        offset += sizeof(compressedMove);
        plain.score = unsignedToSigned((packed.bytes[offset] << 8) | packed.bytes[offset+1]);
        offset += 2;
        std::uint16_t pr = (packed.bytes[offset] << 8) | packed.bytes[offset+1];
        plain.ply = pr & 0x3FFF;
        plain.pos.setPly(plain.ply);
        plain.result = unsignedToSigned(pr >> 14);
        offset += 2;
        plain.pos.setRule50Counter((packed.bytes[offset] << 8) | packed.bytes[offset+1]);

        return plain;
    }

}