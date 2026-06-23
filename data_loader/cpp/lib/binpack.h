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
#include <cassert>
#include <stdexcept>
#include <cstdint>
#include <ios>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <climits>
#include <ctime>
#include <optional>

#include "chess.h"
#include "nodchip.h"
#include "training_data_entry.h"
#include "arithmetic.h"

namespace binpack
{
    constexpr std::size_t KiB = 1024;
    constexpr std::size_t MiB = (1024*KiB);
    constexpr std::size_t GiB = (1024*MiB);

    constexpr std::size_t suggestedChunkSize = MiB;
    constexpr std::size_t maxMovelistSize = 10*KiB; // a safe upper bound
    constexpr std::size_t maxChunkSize = 100*MiB; // to prevent malformed files from causing huge allocations

    inline std::ifstream::pos_type filesize(const char* filename)
    {
        std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
        return in.tellg();
    }

    struct CompressedTrainingDataFile
    {
        struct Header
        {
            std::uint32_t chunkSize;
        };

        CompressedTrainingDataFile(std::string path, std::ios_base::openmode om) :
            m_path(std::move(path)),
            m_file(m_path, std::ios_base::binary | om)
        {
            // Racey but who cares
            m_sizeBytes = filesize(m_path.c_str());
        }

        void append(const char* data, std::uint32_t size)
        {
            writeChunkHeader({size});
            m_file.write(data, size);
            m_sizeBytes += size + 8;
        }

        [[nodiscard]] bool hasNextChunk()
        {
            if (!m_file)
            {
                return false;
            }

            m_file.peek();
            return !m_file.eof();
        }

        void seek_to_start()
        {
            m_file.clear();
            m_file.seekg(0, std::ios_base::beg);
        }

        [[nodiscard]] bool skipChunks(std::size_t n, std::size_t* skipped_out)
        {
            if (skipped_out) *skipped_out = 0;
            if (n == 0) return true;

            std::size_t skipped = 0;

            while (skipped < n)
            {
                if (!hasNextChunk())
                {
                    if (skipped_out) *skipped_out = skipped;
                    return false;
                }

                auto curPos = m_file.tellg();
                Header header = readChunkHeader();
                m_file.seekg(header.chunkSize, std::ios_base::cur);

                assert(m_file.tellg() > curPos);

                ++skipped;
            }

            if (skipped_out) *skipped_out = skipped;
            return true;
        }

        [[nodiscard]] bool skipChunks(std::size_t n)
        {
            return skipChunks(n, nullptr);
        }


        [[nodiscard]] std::vector<unsigned char> readNextChunk()
        {
            auto size = readChunkHeader().chunkSize;
            std::vector<unsigned char> data(size);
            m_file.read(reinterpret_cast<char*>(data.data()), size);
            return data;
        }

        [[nodiscard]] std::size_t sizeBytes() const
        {
            return m_sizeBytes;
        }

        [[nodiscard]] std::string path() const
        {
            return m_path;
        }

    private:
        std::string m_path;
        std::fstream m_file;
        std::size_t m_sizeBytes;

        void writeChunkHeader(Header h)
        {
            unsigned char header[8];
            header[0] = 'B';
            header[1] = 'I';
            header[2] = 'N';
            header[3] = 'P';
            header[4] = h.chunkSize;
            header[5] = h.chunkSize >> 8;
            header[6] = h.chunkSize >> 16;
            header[7] = h.chunkSize >> 24;
            m_file.write(reinterpret_cast<const char*>(header), 8);
        }

        [[nodiscard]] Header readChunkHeader()
        {
            unsigned char header[8];
            m_file.read(reinterpret_cast<char*>(header), 8);
            if (header[0] != 'B' || header[1] != 'I' || header[2] != 'N' || header[3] != 'P')
            {
                throw std::runtime_error("Invalid binpack file or chunk.");
            }

            const std::uint32_t size =
                header[4]
                | (header[5] << 8)
                | (header[6] << 16)
                | (header[7] << 24);

            if (size > maxChunkSize)
            {
                 throw std::runtime_error("Chunk size larger than supported. Malformed file?");
            }

            return { size };
        }
    };


    [[nodiscard]] inline TrainingDataEntry packedSfenValueToTrainingDataEntry(const nodchip::PackedSfenValue& psv)
    {
        TrainingDataEntry ret;

        ret.pos = nodchip::pos_from_packed_sfen(psv.sfen);
        ret.move = psv.move.toMove();
        ret.score = psv.score;
        ret.ply = psv.gamePly;
        ret.result = psv.game_result;

        return ret;
    }

    [[nodiscard]] inline nodchip::PackedSfenValue trainingDataEntryToPackedSfenValue(const TrainingDataEntry& plain)
    {
        nodchip::PackedSfenValue ret;

        nodchip::SfenPacker sp;
        sp.data = reinterpret_cast<uint8_t*>(&ret.sfen);
        sp.pack(plain.pos);

        ret.score = plain.score;
        ret.move = nodchip::StockfishMove::fromMove(plain.move);
        ret.gamePly = plain.ply;
        ret.game_result = plain.result;
        ret.padding = 0xff; // for consistency with the .bin format.

        return ret;
    }

    [[nodiscard]] inline std::size_t usedBitsSafe(std::size_t value)
    {
        if (value == 0) return 0;
        return chess::util::usedBits(value - 1);
    }

    static constexpr std::size_t scoreVleBlockSize = 4;

    struct PackedMoveScoreListReader
    {
        TrainingDataEntry entry;
        std::uint16_t numPlies;
        unsigned char* movetext;

        PackedMoveScoreListReader(const TrainingDataEntry& entry_, unsigned char* movetext_, std::uint16_t numPlies_) :
            entry(entry_),
            numPlies(numPlies_),
            movetext(movetext_),
            m_lastScore(-entry_.score)
        {

        }

        [[nodiscard]] std::uint8_t extractBitsLE8(std::size_t count)
        {
            if (count == 0) return 0;

            if (m_readBitsLeft == 0)
            {
                m_readOffset += 1;
                m_readBitsLeft = 8;
            }

            const std::uint8_t byte = movetext[m_readOffset] << (8 - m_readBitsLeft);
            std::uint8_t bits = byte >> (8 - count);

            if (count > m_readBitsLeft)
            {
                const auto spillCount = count - m_readBitsLeft;
                bits |= movetext[m_readOffset + 1] >> (8 - spillCount);

                m_readBitsLeft += 8;
                m_readOffset += 1;
            }

            m_readBitsLeft -= count;

            return bits;
        }

        [[nodiscard]] std::uint16_t extractVle16(std::size_t blockSize)
        {
            auto mask = (1 << blockSize) - 1;
            std::uint16_t v = 0;
            std::size_t offset = 0;
            for(;;)
            {
                std::uint16_t block = extractBitsLE8(blockSize + 1);
                v |= ((block & mask) << offset);
                if (!(block >> blockSize))
                {
                    break;
                }

                offset += blockSize;
            }
            return v;
        }

        [[nodiscard]] TrainingDataEntry nextEntry()
        {
            entry.pos.doMove(entry.move);
            auto [move, score] = nextMoveScore(entry.pos);
            entry.move = move;
            entry.score = score;
            entry.ply += 1;
            entry.result = -entry.result;
            return entry;
        }

        [[nodiscard]] bool hasNext() const
        {
            return m_numReadPlies < numPlies;
        }

        [[nodiscard]] std::pair<chess::Move, std::int16_t> nextMoveScore(const chess::Position& pos)
        {
            chess::Move move;
            std::int16_t score;

            const chess::Color sideToMove = pos.sideToMove();
            const chess::Bitboard ourPieces = pos.piecesBB(sideToMove);
            const chess::Bitboard theirPieces = pos.piecesBB(!sideToMove);
            const chess::Bitboard occupied = ourPieces | theirPieces;

            const auto pieceId = extractBitsLE8(usedBitsSafe(ourPieces.count()));
            const auto from = chess::Square(chess::nthSetBitIndex(ourPieces.bits(), pieceId));

            const auto pt = pos.pieceAt(from).type();
            switch (pt)
            {
            case chess::PieceType::Pawn:
            {
                const chess::Rank promotionRank = pos.sideToMove() == chess::Color::White ? chess::rank7 : chess::rank2;
                const chess::Rank startRank = pos.sideToMove() == chess::Color::White ? chess::rank2 : chess::rank7;
                const auto forward = sideToMove == chess::Color::White ? chess::FlatSquareOffset(0, 1) : chess::FlatSquareOffset(0, -1);

                const chess::Square epSquare = pos.epSquare();

                chess::Bitboard attackTargets = theirPieces;
                if (epSquare != chess::Square::none())
                {
                    attackTargets |= epSquare;
                }

                chess::Bitboard destinations = chess::bb::pawnAttacks(chess::Bitboard::square(from), sideToMove) & attackTargets;

                const chess::Square sqForward = from + forward;
                if (!occupied.isSet(sqForward))
                {
                    destinations |= sqForward;
                    if (
                        from.rank() == startRank
                        && !occupied.isSet(sqForward + forward)
                        )
                    {
                        destinations |= sqForward + forward;
                    }
                }

                const auto destinationsCount = destinations.count();
                if (from.rank() == promotionRank)
                {
                    const auto moveId = extractBitsLE8(usedBitsSafe(destinationsCount * 4ull));
                    const chess::Piece promotedPiece = chess::Piece(
                        chess::fromOrdinal<chess::PieceType>(ordinal(chess::PieceType::Knight) + (moveId % 4ull)),
                        sideToMove
                    );
                    const auto to = chess::Square(chess::nthSetBitIndex(destinations.bits(), moveId / 4ull));

                    move = chess::Move::promotion(from, to, promotedPiece);
                    break;
                }
                else
                {
                    auto moveId = extractBitsLE8(usedBitsSafe(destinationsCount));
                    const auto to = chess::Square(chess::nthSetBitIndex(destinations.bits(), moveId));
                    if (to == epSquare)
                    {
                        move = chess::Move::enPassant(from, to);
                        break;
                    }
                    else
                    {
                        move = chess::Move::normal(from, to);
                        break;
                    }
                }
            }
            case chess::PieceType::King:
            {
                const chess::CastlingRights ourCastlingRightsMask =
                    sideToMove == chess::Color::White
                    ? chess::CastlingRights::White
                    : chess::CastlingRights::Black;

                const chess::CastlingRights castlingRights = pos.castlingRights();

                const chess::Bitboard attacks = chess::bb::pseudoAttacks<chess::PieceType::King>(from) & ~ourPieces;
                const std::size_t attacksSize = attacks.count();
                const std::size_t numCastlings = chess::intrin::popcount(ordinal(castlingRights & ourCastlingRightsMask));

                const auto moveId = extractBitsLE8(usedBitsSafe(attacksSize + numCastlings));

                if (moveId >= attacksSize)
                {
                    const std::size_t idx = moveId - attacksSize;

                    const chess::CastleType castleType =
                        idx == 0
                        && chess::contains(castlingRights, chess::CastlingTraits::castlingRights[sideToMove][chess::CastleType::Long])
                        ? chess::CastleType::Long
                        : chess::CastleType::Short;

                    move = chess::Move::castle(castleType, sideToMove);
                    break;
                }
                else
                {
                    auto to = chess::Square(chess::nthSetBitIndex(attacks.bits(), moveId));
                    move = chess::Move::normal(from, to);
                    break;
                }
                break;
            }
            default:
            {
                const chess::Bitboard attacks = chess::bb::attacks(pt, from, occupied) & ~ourPieces;
                const auto moveId = extractBitsLE8(usedBitsSafe(attacks.count()));
                auto to = chess::Square(chess::nthSetBitIndex(attacks.bits(), moveId));
                move = chess::Move::normal(from, to);
                break;
            }
            }

            score = m_lastScore + unsignedToSigned(extractVle16(scoreVleBlockSize));
            m_lastScore = -score;

            ++m_numReadPlies;

            return {move, score};
        }

        [[nodiscard]] std::size_t numReadBytes()
        {
            return m_readOffset + (m_readBitsLeft != 8);
        }

    private:
        std::size_t m_readBitsLeft = 8;
        std::size_t m_readOffset = 0;
        std::int16_t m_lastScore = 0;
        std::uint16_t m_numReadPlies = 0;
    };

    struct PackedMoveScoreList
    {
        std::uint16_t numPlies = 0;
        std::vector<unsigned char> movetext;

        void clear(const TrainingDataEntry& e)
        {
            numPlies = 0;
            movetext.clear();
            m_bitsLeft = 0;
            m_lastScore = -e.score;
        }

        void addBitsLE8(std::uint8_t bits, std::size_t count)
        {
            if (count == 0) return;

            if (m_bitsLeft == 0)
            {
                movetext.emplace_back(bits << (8 - count));
                m_bitsLeft = 8;
            }
            else if (count <= m_bitsLeft)
            {
                movetext.back() |= bits << (m_bitsLeft - count);
            }
            else
            {
                const auto spillCount = count - m_bitsLeft;
                movetext.back() |= bits >> spillCount;
                movetext.emplace_back(bits << (8 - spillCount));
                m_bitsLeft += 8;
            }

            m_bitsLeft -= count;
        }

        void addBitsVle16(std::uint16_t v, std::size_t blockSize)
        {
            auto mask = (1 << blockSize) - 1;
            for(;;)
            {
                std::uint8_t block = (v & mask) | ((v > mask) << blockSize);
                addBitsLE8(block, blockSize + 1);
                v >>= blockSize;
                if (v == 0) break;
            }
        }

        void addMoveScore(const chess::Position& pos, chess::Move move, std::int16_t score)
        {
            const chess::Color sideToMove = pos.sideToMove();
            const chess::Bitboard ourPieces = pos.piecesBB(sideToMove);
            const chess::Bitboard theirPieces = pos.piecesBB(!sideToMove);
            const chess::Bitboard occupied = ourPieces | theirPieces;

            const std::uint8_t pieceId = (pos.piecesBB(sideToMove) & chess::bb::before(move.from)).count();
            std::size_t numMoves = 0;
            int moveId = 0;
            const auto pt = pos.pieceAt(move.from).type();
            switch (pt)
            {
            case chess::PieceType::Pawn:
            {
                const chess::Rank secondToLastRank = pos.sideToMove() == chess::Color::White ? chess::rank7 : chess::rank2;
                const chess::Rank startRank = pos.sideToMove() == chess::Color::White ? chess::rank2 : chess::rank7;
                const auto forward = sideToMove == chess::Color::White ? chess::FlatSquareOffset(0, 1) : chess::FlatSquareOffset(0, -1);

                const chess::Square epSquare = pos.epSquare();

                chess::Bitboard attackTargets = theirPieces;
                if (epSquare != chess::Square::none())
                {
                    attackTargets |= epSquare;
                }

                chess::Bitboard destinations = chess::bb::pawnAttacks(chess::Bitboard::square(move.from), sideToMove) & attackTargets;

                const chess::Square sqForward = move.from + forward;
                if (!occupied.isSet(sqForward))
                {
                    destinations |= sqForward;

                    if (
                        move.from.rank() == startRank
                        && !occupied.isSet(sqForward + forward)
                        )
                    {
                        destinations |= sqForward + forward;
                    }
                }

                moveId = (destinations & chess::bb::before(move.to)).count();
                numMoves = destinations.count();
                if (move.from.rank() == secondToLastRank)
                {
                    const auto promotionIndex = (ordinal(move.promotedPiece.type()) - ordinal(chess::PieceType::Knight));
                    moveId = moveId * 4 + promotionIndex;
                    numMoves *= 4;
                }

                break;
            }
            case chess::PieceType::King:
            {
                const chess::CastlingRights ourCastlingRightsMask =
                    sideToMove == chess::Color::White
                    ? chess::CastlingRights::White
                    : chess::CastlingRights::Black;

                const chess::CastlingRights castlingRights = pos.castlingRights();

                const chess::Bitboard attacks = chess::bb::pseudoAttacks<chess::PieceType::King>(move.from) & ~ourPieces;
                const auto attacksSize = attacks.count();
                const auto numCastlingRights = chess::intrin::popcount(ordinal(castlingRights & ourCastlingRightsMask));

                numMoves += attacksSize;
                numMoves += numCastlingRights;

                if (move.type == chess::MoveType::Castle)
                {
                    const auto longCastlingRights = chess::CastlingTraits::castlingRights[sideToMove][chess::CastleType::Long];

                    moveId = attacksSize - 1;

                    if (chess::contains(castlingRights, longCastlingRights))
                    {
                        // We have to add one no matter if it's the used one or not.
                        moveId += 1;
                    }

                    if (chess::CastlingTraits::moveCastlingType(move) == chess::CastleType::Short)
                    {
                        moveId += 1;
                    }
                }
                else
                {
                    moveId = (attacks & chess::bb::before(move.to)).count();
                }
                break;
            }
            default:
            {
                const chess::Bitboard attacks = chess::bb::attacks(pt, move.from, occupied) & ~ourPieces;

                moveId = (attacks & chess::bb::before(move.to)).count();
                numMoves = attacks.count();
            }
            }

            const std::size_t numPieces = ourPieces.count();
            addBitsLE8(pieceId, usedBitsSafe(numPieces));
            addBitsLE8(moveId, usedBitsSafe(numMoves));

            std::uint16_t scoreDelta = signedToUnsigned(score - m_lastScore);
            addBitsVle16(scoreDelta, scoreVleBlockSize);
            m_lastScore = -score;

            ++numPlies;
        }

    private:
        std::size_t m_bitsLeft = 0;
        std::int16_t m_lastScore = 0;
    };

    struct CompressedTrainingDataEntryWriter
    {
        static constexpr std::size_t chunkSize = suggestedChunkSize;

        CompressedTrainingDataEntryWriter(std::string path, std::ios_base::openmode om = std::ios_base::app) :
            m_outputFile(path, om | std::ios_base::out),
            m_lastEntry{},
            m_movelist{},
            m_packedSize(0),
            m_packedEntries(chunkSize + maxMovelistSize),
            m_isFirst(true)
        {
            m_lastEntry.ply = 0xFFFF; // so it's never a continuation
            m_lastEntry.result = 0x7FFF;
        }

        void addTrainingDataEntry(const TrainingDataEntry& e)
        {
            bool isCont = isContinuation(m_lastEntry, e);
            if (isCont)
            {
                // add to movelist
                m_movelist.addMoveScore(e.pos, e.move, e.score);
            }
            else
            {
                if (!m_isFirst)
                {
                    writeMovelist();
                }

                if (m_packedSize >= chunkSize)
                {
                    m_outputFile.append(m_packedEntries.data(), m_packedSize);
                    m_packedSize = 0;
                }

                auto packed = packEntry(e);
                std::memcpy(m_packedEntries.data() + m_packedSize, &packed, sizeof(PackedTrainingDataEntry));
                m_packedSize += sizeof(PackedTrainingDataEntry);

                m_movelist.clear(e);

                m_isFirst = false;
            }

            m_lastEntry = e;
        }

        ~CompressedTrainingDataEntryWriter()
        {
            if (m_packedSize > 0)
            {
                if (!m_isFirst)
                {
                    writeMovelist();
                }

                m_outputFile.append(m_packedEntries.data(), m_packedSize);
                m_packedSize = 0;
            }
        }

    private:
        CompressedTrainingDataFile m_outputFile;
        TrainingDataEntry m_lastEntry;
        PackedMoveScoreList m_movelist;
        std::size_t m_packedSize;
        std::vector<char> m_packedEntries;
        bool m_isFirst;

        void writeMovelist()
        {
            m_packedEntries[m_packedSize++] = m_movelist.numPlies >> 8;
            m_packedEntries[m_packedSize++] = m_movelist.numPlies;
            if (m_movelist.numPlies > 0)
            {
                std::memcpy(m_packedEntries.data() + m_packedSize, m_movelist.movetext.data(), m_movelist.movetext.size());
                m_packedSize += m_movelist.movetext.size();
            }
        };
    };

    struct CompressedTrainingDataEntryReader
    {
        static constexpr std::size_t chunkSize = suggestedChunkSize;

        CompressedTrainingDataEntryReader(std::string path, std::ios_base::openmode om = std::ios_base::app) :
            m_inputFile(path, om | std::ios_base::in),
            m_chunk(),
            m_movelistReader(std::nullopt),
            m_offset(0),
            m_isEnd(false)
        {
            if (!m_inputFile.hasNextChunk())
            {
                m_isEnd = true;
            }
            else
            {
                m_chunk = m_inputFile.readNextChunk();
            }
        }

        [[nodiscard]] bool hasNext()
        {
            return !m_isEnd;
        }

        [[nodiscard]] TrainingDataEntry next()
        {
            if (m_movelistReader.has_value())
            {
                const auto e = m_movelistReader->nextEntry();

                if (!m_movelistReader->hasNext())
                {
                    m_offset += m_movelistReader->numReadBytes();
                    m_movelistReader.reset();

                    fetchNextChunkIfNeeded();
                }

                return e;
            }

            PackedTrainingDataEntry packed;
            std::memcpy(&packed, m_chunk.data() + m_offset, sizeof(PackedTrainingDataEntry));
            m_offset += sizeof(PackedTrainingDataEntry);

            const std::uint16_t numPlies = (m_chunk[m_offset] << 8) | m_chunk[m_offset + 1];
            m_offset += 2;

            const auto e = unpackEntry(packed);

            if (numPlies > 0)
            {
                m_movelistReader.emplace(e, reinterpret_cast<unsigned char*>(m_chunk.data()) + m_offset, numPlies);
            }
            else
            {
                fetchNextChunkIfNeeded();
            }

            return e;
        }

    private:
        CompressedTrainingDataFile m_inputFile;
        std::vector<unsigned char> m_chunk;
        std::optional<PackedMoveScoreListReader> m_movelistReader;
        std::size_t m_offset;
        bool m_isEnd;

        void fetchNextChunkIfNeeded()
        {
            if (m_offset + sizeof(PackedTrainingDataEntry) + 2 > m_chunk.size())
            {
                if (m_inputFile.hasNextChunk())
                {
                    m_chunk = m_inputFile.readNextChunk();
                    m_offset = 0;
                }
                else
                {
                    m_isEnd = true;
                }
            }
        }
    };
}