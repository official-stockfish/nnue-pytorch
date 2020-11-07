#ifndef _SFEN_STREAM_H_
#define _SFEN_STREAM_H_

#include "nnue_data_binpack_format.h"

#include <optional>
#include <fstream>
#include <string>
#include <memory>

namespace training_data {

    using namespace binpack;

    static bool ends_with(const std::string& lhs, const std::string& end)
    {
        if (end.size() > lhs.size()) return false;

        return std::equal(end.rbegin(), end.rend(), lhs.rbegin());
    }

    static bool has_extension(const std::string& filename, const std::string& extension)
    {
        return ends_with(filename, "." + extension);
    }

    static std::string filename_with_extension(const std::string& filename, const std::string& ext)
    {
        if (ends_with(filename, ext))
        {
            return filename;
        }
        else
        {
            return filename + "." + ext;
        }
    }

    struct BasicSfenInputStream
    {
        virtual std::optional<TrainingDataEntry> next() = 0;
        virtual bool eof() const = 0;
        virtual ~BasicSfenInputStream() {}
    };

    struct BinSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "bin";

        BinSfenInputStream(std::string filename) :
            m_stream(filename, openmode),
            m_eof(!m_stream)
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            nodchip::PackedSfenValue e;
            if(m_stream.read(reinterpret_cast<char*>(&e), sizeof(nodchip::PackedSfenValue)))
            {
                return packedSfenValueToTrainingDataEntry(e);
            }
            else
            {
                m_eof = true;
                return std::nullopt;
            }
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinSfenInputStream() override {}

    private:
        std::fstream m_stream;
        bool m_eof;
    };

    struct BinpackSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "binpack";

        BinpackSfenInputStream(std::string filename) :
            m_stream(filename, openmode),
            m_eof(!m_stream.hasNext())
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            if (!m_stream.hasNext())
            {
                m_eof = true;
                return std::nullopt;
            }

            return m_stream.next();
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinpackSfenInputStream() override {}

    private:
        binpack::CompressedTrainingDataEntryReader m_stream;
        bool m_eof;
    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file(const std::string& filename)
    {
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename);
        else if (has_extension(filename, BinpackSfenInputStream::extension))
            return std::make_unique<BinpackSfenInputStream>(filename);

        return nullptr;
    }
}

#endif