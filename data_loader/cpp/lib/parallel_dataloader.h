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

#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cassert>
#include <ios>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <climits>
#include <ctime>
#include <optional>
#include <thread>
#include <mutex>
#include <random>
#include <functional>
#include <type_traits>
#include <chrono>

#include "rng.h"
#include "thread_safe_types.h"
#include "binpack.h"
#include "training_data_entry.h"


namespace binpack
{
    using namespace std::literals;

    struct CompressedTrainingDataEntryParallelReader
    {
        static constexpr std::size_t chunkSize = suggestedChunkSize;
        static constexpr std::size_t sharedChunkQueueCapacity = 256;
        using FileChunk = std::vector<unsigned char>;

        CompressedTrainingDataEntryParallelReader(
            int concurrency,
            std::vector<std::string> paths,
            std::ios_base::openmode om = std::ios_base::app,
            bool cyclic = false,
            std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr,
            int rank = 0,
            int world_size = 1
        ) :
            m_concurrency(concurrency),
            m_numRunningWorkers(concurrency),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate)),
            m_rank(rank),
            m_world_size(world_size)
        {
            std::vector<double> sizes; // discrete distribution wants double weights
            for (const auto& path : paths)
            {
                auto& file = m_inputFiles.emplace_back(path, om | std::ios_base::in);

                if (!file.hasNextChunk()) [[unlikely]]
                {
                     throw std::runtime_error("Empty or corrupted file: " + path);
                }

                sizes.emplace_back(static_cast<double>(file.sizeBytes()));
            }

            for (size_t i = 0; i < m_inputFiles.size(); ++i)
            {
                m_fileMutexes.push_back(std::make_unique<std::timed_mutex>());
            }
            m_distribution_weights = sizes;
            m_ringBuffer.reserve_internal(threadBufferSize);

            // Initialize DDP seeking tracking
            m_files_seeked_for_ddp.resize(m_inputFiles.size(), false);
            m_ddp_chunks_to_skip_after_read.resize(m_inputFiles.size(), 0);

            m_stopFlag.store(false);
            m_readersFinished.store(false);

            int numReaders = std::max(1, static_cast<int>(std::min(0.5 * paths.size(), 0.5 * concurrency)));
            m_numRunningReaders.store(numReaders);

            m_fileExhausted = std::make_unique<std::atomic_bool[]>(m_inputFiles.size());
            for (size_t i = 0; i < m_inputFiles.size(); ++i)
            {
                m_fileExhausted[i].store(false, std::memory_order_relaxed);
            }

            auto readerWorker = [this]()
            {
                auto& prng = rng::get_thread_local_rng();
                std::discrete_distribution<std::size_t> local_dist(
                    m_distribution_weights.begin(), m_distribution_weights.end()
                );

                while (!m_stopFlag.load())
                {
                    bool allExhausted = true;
                    for (size_t i = 0; i < m_inputFiles.size(); ++i)
                    {
                        if (!m_fileExhausted[i].load(std::memory_order_relaxed))
                        {
                            allExhausted = false;
                            break;
                        }
                    }

                    if (allExhausted)
                    {
                        break;
                    }

                    std::size_t fileId = local_dist(prng);

                    if (m_fileExhausted[fileId].load(std::memory_order_relaxed))
                    {
                        continue;
                    }

                    // Try to lock the file mutex
                    std::unique_lock lock(*m_fileMutexes[fileId], std::defer_lock);
                    if (!lock.try_lock_for(kMaxLockWaitTime))
                    {
                        m_timeout_count.fetch_add(1, std::memory_order_relaxed);

                        auto now = std::chrono::steady_clock::now().time_since_epoch();
                        int64_t now_sec = std::chrono::duration_cast<std::chrono::seconds>(now).count();
                        int64_t last_sec = m_last_warning_time.load(std::memory_order_relaxed);

                        if (now_sec - last_sec >= kWarningCooldownSeconds)
                        {
                            if (m_last_warning_time.compare_exchange_strong(last_sec, now_sec, std::memory_order_relaxed))
                            {
                                uint64_t count_to_print = m_timeout_count.exchange(0, std::memory_order_relaxed);

                                auto utc_now = std::chrono::system_clock::now();
                                std::time_t utc_time = std::chrono::system_clock::to_time_t(utc_now);

                                auto to_utc_tm = [](std::time_t time, std::tm& result)
                                {
                                    #if defined(_MSC_VER)
                                    gmtime_s(&result, &time);
                                    #else
                                    gmtime_r(&time, &result);
                                    #endif
                                };

                                std::tm utc_tm{};
                                to_utc_tm(utc_time, utc_tm);

                                std::cerr << "[" << std::put_time(&utc_tm, "%Y-%m-%d %H:%M:%S UTC") << "] "
                                          << "[Warning] Dataloader mutex acquisition for file with ID "
                                          << fileId << " name " << m_inputFiles[fileId].path()
                                          << " timed out after " << kMaxLockWaitTime.count()
                                          << "ms. Re-rolling file. "
                                          << "(" << count_to_print << " timeouts since last warning)\n";
                            }
                        }
                        continue;
                    }

                    auto& inputFile = m_inputFiles[fileId];

                    auto seek_for_ddp_rank = [&](std::size_t rank) -> bool
                    {
                        std::size_t skipped = 0;
                        if (inputFile.skipChunks(rank, &skipped))
                        {
                            return true;
                        }
                        if (!m_cyclic)
                        {
                            return false;
                        }
                        if (skipped == 0)
                        {
                            return false;
                        }
                        inputFile.seek_to_start();
                        const std::size_t offset = rank % skipped;
                        const bool ok = inputFile.skipChunks(offset);
                        assert(ok);
                        return ok;
                    };

                    // DDP: chunk-based skipping
                    if (m_world_size > 1)
                    {
                        if (!m_files_seeked_for_ddp[fileId])
                        {
                            const std::size_t rank = static_cast<std::size_t>(m_rank);
                            if (!seek_for_ddp_rank(rank))
                            {
                                m_fileExhausted[fileId].store(true, std::memory_order_relaxed);
                                continue;
                            }
                            m_files_seeked_for_ddp[fileId] = true;
                        }
                        else if (m_ddp_chunks_to_skip_after_read[fileId] > 0)
                        {
                            const bool success = inputFile.skipChunks(m_ddp_chunks_to_skip_after_read[fileId]);
                            if (!success)
                            {
                                if (!m_cyclic)
                                {
                                    m_fileExhausted[fileId].store(true, std::memory_order_relaxed);
                                    continue;
                                }
                                inputFile.seek_to_start();
                                const std::size_t rank = static_cast<std::size_t>(m_rank);
                                if (!seek_for_ddp_rank(rank))
                                {
                                    m_fileExhausted[fileId].store(true, std::memory_order_relaxed);
                                    continue;
                                }
                            }
                            m_ddp_chunks_to_skip_after_read[fileId] = 0;
                        }
                    }

                    if (!inputFile.hasNextChunk())
                    {
                        if (m_cyclic)
                        {
                            inputFile.seek_to_start();

                            if (m_world_size > 1)
                            {
                                const std::size_t rank = static_cast<std::size_t>(m_rank);
                                if (!seek_for_ddp_rank(rank))
                                {
                                    m_fileExhausted[fileId].store(true, std::memory_order_relaxed);
                                    continue;
                                }
                            }
                        }
                        else
                        {
                            m_fileExhausted[fileId].store(true, std::memory_order_relaxed);
                            continue;
                        }
                    }

                    std::vector<unsigned char> chunk = inputFile.readNextChunk();

                    if (m_world_size > 1)
                    {
                        m_ddp_chunks_to_skip_after_read[fileId] = static_cast<std::size_t>(m_world_size - 1);
                    }

                    lock.unlock(); // Release file lock immediately after read

                    bool success = m_sharedChunkQueue.put(chunk, [this]() {
                        return this->m_stopFlag.load();
                    });

                    if (!success)
                    {
                        break;
                    }
                }

                if (m_numRunningReaders.fetch_sub(1) == 1)
                {
                    m_readersFinished.store(true);
                    m_sharedChunkQueue.signal_stop();
                }
            };

            for (int i = 0; i < numReaders; ++i)
            {
                m_readerThreads.emplace_back(readerWorker);
            }

            auto worker = [this]()
            {
                std::vector<unsigned char> m_chunk{};
                std::optional<PackedMoveScoreListReader> m_movelistReader(std::nullopt);
                std::size_t m_offset(0);
                std::vector<TrainingDataEntry> m_localBuffer;
                m_localBuffer.reserve(threadBufferSize);

                bool isEnd = fetchNextChunkFromSharedQueue(m_offset, m_chunk);

                while(!isEnd && !m_stopFlag.load())
                {
                    while (m_localBuffer.size() < threadBufferSize)
                    {
                        if (m_movelistReader.has_value())
                        {
                            const auto e = m_movelistReader->nextEntry();

                            if (!m_movelistReader->hasNext())
                            {
                                m_offset += m_movelistReader->numReadBytes();
                                m_movelistReader.reset();

                                isEnd = fetchNextChunkFromSharedQueue(m_offset, m_chunk);
                            }

                            if (!m_skipPredicate || !m_skipPredicate(e))
                                m_localBuffer.emplace_back(e);
                        }
                        else
                        {
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
                                isEnd = fetchNextChunkFromSharedQueue(m_offset, m_chunk);
                            }

                            if (!m_skipPredicate || !m_skipPredicate(e))
                                m_localBuffer.emplace_back(e);
                        }

                        if (isEnd || m_stopFlag.load())
                        {
                            break;
                        }
                    }

                    if (!m_localBuffer.empty())
                    {
                        auto& prng = rng::get_thread_local_rng();
                        std::shuffle(m_localBuffer.begin(), m_localBuffer.end(), prng);

                        bool success = m_ringBuffer.put(m_localBuffer, [this]() {
                            return this->should_stop_producer();
                        });
                        if (!success) break; // Ring and workers exhausted

                        m_localBuffer.clear();
                        m_localBuffer.reserve(threadBufferSize);
                    }
                }
                m_numRunningWorkers.fetch_sub(1);
                m_ringBuffer.signal_stop(false);
            };

            for (int i = 0; i < concurrency; ++i)
            {
                m_workers.emplace_back(worker);
            }
        }

        [[nodiscard]] std::optional<TrainingDataEntry> next()
        {
            LocalBuffer& local = m_bufferRegistry.get();
            if (local.offset < local.entries.size()) [[likely]]
            {
                return std::move(local.entries[local.offset++]);
            }

            if (local.offset >= local.entries.size())
            {
                bool success = m_ringBuffer.take(local.entries, [this]() {
                    return this->should_stop_consumer();
                });
                if (!success) return std::nullopt;
                local.offset = 0;
            }

            return std::move(local.entries[local.offset++]);
        }

        int fill(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            LocalBuffer& local = m_bufferRegistry.get();
            std::size_t total_filled = 0;

            while (total_filled < n) {
                if (local.offset >= local.entries.size()) [[unlikely]]
                {
                    bool success = m_ringBuffer.take(local.entries, [this]() {
                        return this->should_stop_consumer();
                    });
                    if (!success) break; // Ring and workers exhausted
                    local.offset = 0;
                }

                const std::size_t available = local.entries.size() - local.offset;
                const std::size_t to_copy = std::min(n - total_filled, available);

                vec.insert(
                    vec.end(),
                    std::make_move_iterator(local.entries.begin() + local.offset),
                    std::make_move_iterator(local.entries.begin() + local.offset + to_copy)
                );

                local.offset += to_copy;
                total_filled += to_copy;
            }
            return static_cast<int>(total_filled);
        }

        ~CompressedTrainingDataEntryParallelReader()
        {
            m_stopFlag.store(true);
            m_sharedChunkQueue.signal_stop();
            m_ringBuffer.signal_stop();
            for (auto& reader : m_readerThreads)
            {
                if (reader.joinable())
                {
                    reader.join();
                }
            }
            for (auto& worker : m_workers)
            {
                if (worker.joinable())
                {
                    worker.join();
                }
            }
        }

    private:
        int m_concurrency;
        std::atomic_int m_numRunningWorkers;
        std::vector<CompressedTrainingDataFile> m_inputFiles;
        bool m_cyclic;

        static constexpr int threadBufferSize = 256 * 256 * 16;

        std::atomic_bool m_stopFlag;
        std::vector<std::thread> m_workers;

        // Per File Lock
        std::vector<std::unique_ptr<std::timed_mutex>> m_fileMutexes;
        std::vector<double> m_distribution_weights;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;

        // Avoid blocking too long on a contended per-file mutex; if locking times out,
        // the worker can retry by selecting a different file, and warnings are rate-limited.
        // This is especially important if one fileserver is particularly slow.
        static constexpr std::chrono::milliseconds kMaxLockWaitTime{2000};
        static constexpr int64_t kWarningCooldownSeconds = 300;
        std::atomic<uint64_t> m_timeout_count{0};
        std::atomic<int64_t> m_last_warning_time{-kWarningCooldownSeconds};

        // DDP support
        int m_rank;
        int m_world_size;
        std::vector<std::uint8_t> m_files_seeked_for_ddp;  // Track which files have been seeked for DDP
        std::vector<std::size_t> m_ddp_chunks_to_skip_after_read;

        // thread local data buffers
        using TrainingDataEntries = std::vector<TrainingDataEntry>;
        struct alignas(128) LocalBuffer {
            TrainingDataEntries entries;
            size_t offset = 0;
        };

        // Constant Size Ring Buffer
        static constexpr int ringCapacity = 1;
        thread_safe_types::ThreadLocalRegistry<LocalBuffer> m_bufferRegistry;
        thread_safe_types::ThreadSafeRingBuffer<TrainingDataEntries, ringCapacity> m_ringBuffer;

        bool should_stop_producer()
        {
            return m_stopFlag.load();
        }

        bool should_stop_consumer()
        {
            return m_numRunningWorkers.load() <= 0;
        }

        bool fetchNextChunkFromSharedQueue(std::size_t& m_offset, std::vector<unsigned char>& m_chunk)
        {
            if (m_offset + sizeof(PackedTrainingDataEntry) + 2 > m_chunk.size())
            {
                if (m_stopFlag.load())
                {
                    return true;
                }

                if (m_readersFinished.load() && m_sharedChunkQueue.is_empty())
                {
                    return true;
                }

                bool success = m_sharedChunkQueue.take(
                    m_chunk,
                    [this]() { return m_stopFlag.load() || (m_readersFinished.load() && m_sharedChunkQueue.is_empty()); }
                );

                if (success)
                {
                    m_offset = 0;
                    return false;
                }
                return true;
            }

            return false;
        }

        // Shared Raw Chunk Queue
        thread_safe_types::ThreadSafeRingBuffer<FileChunk, sharedChunkQueueCapacity> m_sharedChunkQueue;

        std::unique_ptr<std::atomic_bool[]> m_fileExhausted;
        std::vector<std::thread> m_readerThreads;
        std::atomic_bool m_readersFinished;
        std::atomic_int m_numRunningReaders;
    };

}
