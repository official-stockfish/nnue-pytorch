#ifndef _SFEN_STREAM_H_
#define _SFEN_STREAM_H_

#include "nnue_training_data_formats.h"

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

        BinSfenInputStream(std::string filename, bool cyclic) :
            m_stream(filename, openmode),
            m_filename(filename),
            m_eof(!m_stream),
            m_cyclic(cyclic)
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
                if (m_cyclic)
                {
                    m_stream = std::fstream(m_filename, openmode);
                    if (!m_stream)
                        return std::nullopt;
                    return next();
                }

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
        std::string m_filename;
        bool m_eof;
        bool m_cyclic;
    };

    struct BinpackSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "binpack";

        BinpackSfenInputStream(std::string filename, bool cyclic) :
            m_stream(std::make_unique<binpack::CompressedTrainingDataEntryReader>(filename, openmode)),
            m_filename(filename),
            m_eof(!m_stream->hasNext()),
            m_cyclic(cyclic)
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            if (!m_stream->hasNext())
            {
                if (m_cyclic)
                {
                    m_stream = std::make_unique<binpack::CompressedTrainingDataEntryReader>(m_filename, openmode);
                    if (!m_stream->hasNext())
                        return std::nullopt;
                    return next();
                }

                m_eof = true;
                return std::nullopt;
            }

            return m_stream->next();
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinpackSfenInputStream() override {}

    private:
        std::unique_ptr<binpack::CompressedTrainingDataEntryReader> m_stream;
        std::string m_filename;
        bool m_eof;
        bool m_cyclic;
    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file(const std::string& filename, bool cyclic)
    {
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic);
        else if (has_extension(filename, BinpackSfenInputStream::extension))
            return std::make_unique<BinpackSfenInputStream>(filename, cyclic);

        return nullptr;
    }
}

#endif