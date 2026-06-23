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

#include <cstdio>
#include <cstdint>
#include <cassert>
#include <string>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <climits>
#include <ctime>


#include "chess.h"


namespace binpack::nodchip
{
    // This namespace contains modified code from https://github.com/nodchip/Stockfish
    // which is released under GPL v3 license https://www.gnu.org/licenses/gpl-3.0.html

    using namespace std;

    struct StockfishMove
    {
        [[nodiscard]] static StockfishMove fromMove(chess::Move move)
        {
            StockfishMove sfm;

            sfm.m_raw = 0;

            unsigned moveFlag = 0;
            if (move.type == chess::MoveType::Promotion) moveFlag = 1;
            else if (move.type == chess::MoveType::EnPassant) moveFlag = 2;
            else if (move.type == chess::MoveType::Castle) moveFlag = 3;

            unsigned promotionIndex = 0;
            if (move.type == chess::MoveType::Promotion)
            {
                promotionIndex = static_cast<int>(move.promotedPiece.type()) - static_cast<int>(chess::PieceType::Knight);
            }

            sfm.m_raw |= static_cast<std::uint16_t>(moveFlag);
            sfm.m_raw <<= 2;
            sfm.m_raw |= static_cast<std::uint16_t>(promotionIndex);
            sfm.m_raw <<= 6;
            sfm.m_raw |= static_cast<int>(move.from);
            sfm.m_raw <<= 6;
            sfm.m_raw |= static_cast<int>(move.to);

            return sfm;
        }

        [[nodiscard]] chess::Move toMove() const
        {
            const chess::Square to = static_cast<chess::Square>((m_raw & (0b111111 << 0) >> 0));
            const chess::Square from = static_cast<chess::Square>((m_raw & (0b111111 << 6)) >> 6);

            const unsigned promotionIndex = (m_raw & (0b11 << 12)) >> 12;
            const chess::PieceType promotionType = static_cast<chess::PieceType>(static_cast<int>(chess::PieceType::Knight) + promotionIndex);

            const unsigned moveFlag = (m_raw & (0b11 << 14)) >> 14;
            chess::MoveType type = chess::MoveType::Normal;
            if (moveFlag == 1) type = chess::MoveType::Promotion;
            else if (moveFlag == 2) type = chess::MoveType::EnPassant;
            else if (moveFlag == 3) type = chess::MoveType::Castle;

            if (type == chess::MoveType::Promotion)
            {
                const chess::Color stm = to.rank() == chess::rank8 ? chess::Color::White : chess::Color::Black;
                return chess::Move{from, to, type, chess::Piece(promotionType, stm)};
            }

            return chess::Move{from, to, type};
        }

        [[nodiscard]] std::string toString() const
        {
            const chess::Square to = static_cast<chess::Square>((m_raw & (0b111111 << 0) >> 0));
            const chess::Square from = static_cast<chess::Square>((m_raw & (0b111111 << 6)) >> 6);

            const unsigned promotionIndex = (m_raw & (0b11 << 12)) >> 12;
            const chess::PieceType promotionType = static_cast<chess::PieceType>(static_cast<int>(chess::PieceType::Knight) + promotionIndex);

            std::string r;
            chess::parser_bits::appendSquareToString(from, r);
            chess::parser_bits::appendSquareToString(to, r);
            if (promotionType != chess::PieceType::None)
            {
                r += chess::EnumTraits<chess::PieceType>::toChar(promotionType, chess::Color::Black);
            }

            return r;
        }

    private:
        std::uint16_t m_raw;
    };
    static_assert(sizeof(StockfishMove) == sizeof(std::uint16_t));

    struct PackedSfen
    {
        uint8_t data[32];
    };

    struct PackedSfenValue
    {
        // phase
        PackedSfen sfen;

        // Evaluation value returned from Learner::search()
        int16_t score;

        // PV first move
        // Used when finding the match rate with the teacher
        StockfishMove move;

        // Trouble of the phase from the initial phase.
        uint16_t gamePly;

        // 1 if the player on this side ultimately wins the game. -1 if you are losing.
        // 0 if a draw is reached.
        // The draw is in the teacher position generation command gensfen,
        // Only write if LEARN_GENSFEN_DRAW_RESULT is enabled.
        int8_t game_result;

        // When exchanging the file that wrote the teacher aspect with other people
        //Because this structure size is not fixed, pad it so that it is 40 bytes in any environment.
        uint8_t padding;

        // 32 + 2 + 2 + 2 + 1 + 1 = 40bytes
    };
    static_assert(sizeof(PackedSfenValue) == 40);
    // Class that handles bitstream

    // useful when doing aspect encoding
    struct BitStream
    {
        // Set the memory to store the data in advance.
        // Assume that memory is cleared to 0.
        void  set_data(uint8_t* data_) { data = data_; reset(); }

        // Get the pointer passed in set_data().
        uint8_t* get_data() const { return data; }

        // Get the cursor.
        int get_cursor() const { return bit_cursor; }

        // reset the cursor
        void reset() { bit_cursor = 0; }

        // Write 1bit to the stream.
        // If b is non-zero, write out 1. If 0, write 0.
        void write_one_bit(int b)
        {
            if (b)
                data[bit_cursor / 8] |= 1 << (bit_cursor & 7);

            ++bit_cursor;
        }

        // Get 1 bit from the stream.
        int read_one_bit()
        {
            int b = (data[bit_cursor / 8] >> (bit_cursor & 7)) & 1;
            ++bit_cursor;

            return b;
        }

        // write n bits of data
        // Data shall be written out from the lower order of d.
        void write_n_bit(int d, int n)
        {
            for (int i = 0; i <n; ++i)
                write_one_bit(d & (1 << i));
        }

        // read n bits of data
        // Reverse conversion of write_n_bit().
        int read_n_bit(int n)
        {
            int result = 0;
            for (int i = 0; i < n; ++i)
                result |= read_one_bit() ? (1 << i) : 0;

            return result;
        }

    private:
        // Next bit position to read/write.
        int bit_cursor;

        // data entity
        uint8_t* data;
    };


    // Huffman coding
    // * is simplified from mini encoding to make conversion easier.
    //
    // Huffman Encoding
    //
    // Empty  xxxxxxx0
    // Pawn   xxxxx001 + 1 bit (Color)
    // Knight xxxxx011 + 1 bit (Color)
    // Bishop xxxxx101 + 1 bit (Color)
    // Rook   xxxxx111 + 1 bit (Color)
    // Queen   xxxx1001 + 1 bit (Color)
    //
    // Worst case:
    // - 32 empty squares    32 bits
    // - 30 pieces           150 bits
    // - 2 kings             12 bits
    // - castling rights     4 bits
    // - ep square           7 bits
    // - rule50              7 bits
    // - game ply            16 bits
    // - TOTAL               228 bits < 256 bits

    struct HuffmanedPiece
    {
        int code; // how it will be coded
        int bits; // How many bits do you have
    };

    // NOTE: Order adjusted for this library because originally NO_PIECE had index 0
    constexpr HuffmanedPiece huffman_table[] =
    {
        {0b0001,4}, // PAWN     1
        {0b0011,4}, // KNIGHT   3
        {0b0101,4}, // BISHOP   5
        {0b0111,4}, // ROOK     7
        {0b1001,4}, // QUEEN    9
        {-1,-1},    // KING - unused
        {0b0000,1}, // NO_PIECE 0
    };

    // Class for compressing/decompressing sfen
    // sfen can be packed to 256bit (32bytes) by Huffman coding.
    // This is proven by mini. The above is Huffman coding.
    //
    // Internal format = 1-bit turn + 7-bit king position *2 + piece on board (Huffman coding) + hand piece (Huffman coding)
    // Side to move (White = 0, Black = 1) (1bit)
    // White King Position (6 bits)
    // Black King Position (6 bits)
    // Huffman Encoding of the board
    // Castling availability (1 bit x 4)
    // En passant square (1 or 1 + 6 bits)
    // Rule 50 (6 bits)
    // Game play (8 bits)
    //
    // TODO(someone): Rename SFEN to FEN.
    //
    struct SfenPacker
    {
        // Pack sfen and store in data[32].
        void pack(const chess::Position& pos)
        {
            memset(data, 0, 32 /* 256bit */);
            stream.set_data(data);

            // turn
            // Side to move.
            stream.write_one_bit((int)(pos.sideToMove()));

            // 6-bit positions for White and Black Kings
            stream.write_n_bit(static_cast<int>(pos.kingSquare(chess::Color::White)), 6);
            stream.write_n_bit(static_cast<int>(pos.kingSquare(chess::Color::Black)), 6);

            // Write the pieces on the board other than the kings.
            for (chess::Rank r = chess::rank8; r >= chess::rank1; --r)
            {
                for (chess::File f = chess::fileA; f <= chess::fileH; ++f)
                {
                    chess::Piece pc = pos.pieceAt(chess::Square(f, r));
                    if (pc.type() == chess::PieceType::King)
                        continue;
                    write_board_piece_to_stream(pc);
                }
            }

            // TODO(someone): Support chess960.
            auto cr = pos.castlingRights();
            stream.write_one_bit(contains(cr, chess::CastlingRights::WhiteKingSide));
            stream.write_one_bit(contains(cr, chess::CastlingRights::WhiteQueenSide));
            stream.write_one_bit(contains(cr, chess::CastlingRights::BlackKingSide));
            stream.write_one_bit(contains(cr, chess::CastlingRights::BlackQueenSide));

            if (pos.epSquare() == chess::Square::none()) {
                stream.write_one_bit(0);
            }
            else {
                stream.write_one_bit(1);
                stream.write_n_bit(static_cast<int>(pos.epSquare()), 6);
            }

            stream.write_n_bit(pos.rule50Counter(), 6);

            stream.write_n_bit(pos.fullMove(), 8);

            // Write high bits of half move. This is a fix for the
            // limited range of half move counter.
            // This is backwards compatible.
            stream.write_n_bit(pos.fullMove() >> 8, 8);

            // Write the highest bit of rule50 at the end. This is a backwards
            // compatible fix for rule50 having only 6 bits stored.
            // This bit is just ignored by the old parsers.
            stream.write_n_bit(pos.rule50Counter() >> 6, 1);

            assert(stream.get_cursor() <= 256);
        }

        // sfen packed by pack() (256bit = 32bytes)
        // Or sfen to decode with unpack()
        uint8_t *data; // uint8_t[32];

        BitStream stream;

        // Output the board pieces to stream.
        void write_board_piece_to_stream(chess::Piece pc)
        {
            // piece type
            chess::PieceType pr = pc.type();
            auto c = huffman_table[static_cast<int>(pr)];
            stream.write_n_bit(c.code, c.bits);

            if (pc == chess::Piece::none())
                return;

            // first and second flag
            stream.write_one_bit(static_cast<int>(pc.color()));
        }

        // Read one board piece from stream
        [[nodiscard]] chess::Piece read_board_piece_from_stream()
        {
            int pr = static_cast<int>(chess::PieceType::None);
            int code = 0, bits = 0;
            while (true)
            {
                code |= stream.read_one_bit() << bits;
                ++bits;

                assert(bits <= 6);

                for (pr = static_cast<int>(chess::PieceType::Pawn); pr <= static_cast<int>(chess::PieceType::None); ++pr)
                    if (huffman_table[pr].code == code
                        && huffman_table[pr].bits == bits)
                        goto Found;
            }
        Found:;
            if (pr == static_cast<int>(chess::PieceType::None))
                return chess::Piece::none();

            // first and second flag
            chess::Color c = (chess::Color)stream.read_one_bit();

            return chess::Piece(static_cast<chess::PieceType>(pr), c);
        }
    };


    [[nodiscard]] inline chess::Position pos_from_packed_sfen(const PackedSfen& sfen)
    {
        SfenPacker packer;
        auto& stream = packer.stream;
        stream.set_data(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&sfen)));

        chess::Position pos{};

        // Active color
        pos.setSideToMove((chess::Color)stream.read_one_bit());

        // First the position of the Kings
        pos.place(chess::Piece(chess::PieceType::King, chess::Color::White), static_cast<chess::Square>(stream.read_n_bit(6)));
        pos.place(chess::Piece(chess::PieceType::King, chess::Color::Black), static_cast<chess::Square>(stream.read_n_bit(6)));

        // Piece placement
        for (chess::Rank r = chess::rank8; r >= chess::rank1; --r)
        {
            for (chess::File f = chess::fileA; f <= chess::fileH; ++f)
            {
                auto sq = chess::Square(f, r);

                // Check if the square is already occupied by a King
                chess::Piece pc;
                if (pos.pieceAt(sq).type() != chess::PieceType::King)
                {
                    assert(pos.pieceAt(sq) == chess::Piece::none());
                    pc = packer.read_board_piece_from_stream();
                }
                else
                {
                    pc = pos.pieceAt(sq);
                }

                // There may be no pieces, so skip in that case.
                if (pc == chess::Piece::none())
                    continue;

                if (pc.type() != chess::PieceType::King)
                {
                    pos.place(pc, sq);
                }

                assert(stream.get_cursor() <= 256);
            }
        }

        // Castling availability.
        chess::CastlingRights cr = chess::CastlingRights::None;
        if (stream.read_one_bit()) {
            cr |= chess::CastlingRights::WhiteKingSide;
        }
        if (stream.read_one_bit()) {
            cr |= chess::CastlingRights::WhiteQueenSide;
        }
        if (stream.read_one_bit()) {
            cr |= chess::CastlingRights::BlackKingSide;
        }
        if (stream.read_one_bit()) {
            cr |= chess::CastlingRights::BlackQueenSide;
        }
        pos.setCastlingRights(cr);

        // En passant square. Ignore if no pawn capture is possible
        if (stream.read_one_bit()) {
            chess::Square ep_square = static_cast<chess::Square>(stream.read_n_bit(6));
            pos.setEpSquare(ep_square);
        }

        // Halfmove clock
        std::uint8_t rule50 = stream.read_n_bit(6);

        // Fullmove number
        std::uint16_t fullmove = stream.read_n_bit(8);

        // Fullmove number, high bits
        // This was added as a fix for fullmove clock
        // overflowing at 256. This change is backwards compatible.
        fullmove |= stream.read_n_bit(8) << 8;

        // Read the highest bit of rule50. This was added as a fix for rule50
        // counter having only 6 bits stored.
        // In older entries this will just be a zero bit.
        rule50 |= stream.read_n_bit(1) << 6;

        pos.setFullMove(fullmove);
        pos.setRule50Counter(rule50);

        assert(stream.get_cursor() <= 256);

        return pos;
    }
}