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
        virtual void fill(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                auto v = this->next();
                if (!v.has_value())
                {
                    break;
                }
                vec.emplace_back(*v);
            }
        }

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

    struct BinpackSfenInputParallelStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "binpack";

        BinpackSfenInputParallelStream(int concurrency, std::string filename, bool cyclic) :
            m_stream(std::make_unique<binpack::CompressedTrainingDataEntryParallelReader>(concurrency, filename, openmode)),
            m_filename(filename),
            m_concurrency(concurrency),
            m_eof(false),
            m_cyclic(cyclic)
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            auto v = m_stream->next();
            if (!v.has_value())
            {
                if (m_cyclic)
                {
                    m_stream = std::make_unique<binpack::CompressedTrainingDataEntryParallelReader>(m_concurrency, m_filename, openmode);
                    return m_stream->next();
                }

                m_eof = true;
                return std::nullopt;
            }

            return v;
        }

        void fill(std::vector<TrainingDataEntry>& v, std::size_t n) override
        {
            auto k = m_stream->fill(v, n);
            if (n != k)
            {
                if (m_cyclic)
                {
                    m_stream = std::make_unique<binpack::CompressedTrainingDataEntryParallelReader>(m_concurrency, m_filename, openmode);
                    n -= k;
                    k = m_stream->fill(v, n);
                    if (k == 0)
                    {
                        // No data in the file
                        m_eof = true;
                        return;
                    }
                    else if (k == n)
                    {
                        // We're done
                        return;
                    }
                    else
                    {
                        // We need to read again
                        this->fill(v, n - k);
                    }
                }
                else
                {
                    m_eof = true;
                }
            }
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinpackSfenInputParallelStream() override {}

    private:
        std::unique_ptr<binpack::CompressedTrainingDataEntryParallelReader> m_stream;
        std::string m_filename;
        int m_concurrency;
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

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file_parallel(int concurrency, const std::string& filename, bool cyclic)
    {
        // TODO (low priority): optimize and parallelize .bin reading.
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic);
        else if (has_extension(filename, BinpackSfenInputParallelStream::extension))
            return std::make_unique<BinpackSfenInputParallelStream>(concurrency, filename, cyclic);

        return nullptr;
    }
}

#endif
