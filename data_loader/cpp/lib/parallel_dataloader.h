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

/*
 * Data loader pipeline map (one rank; numbers at --num-workers=32
 * with 43 input files).
 *
 *           binpack files on storage: 43 files / 655 GiB (Lustre)
 *           chunk = ~1 MiB blob + 8 B "BINP" header ~= 440k positions
 *           (~2.3 bytes/position on disk)
 *                                  |
 *                                  |  14 readers
 *                                  |  one chunk per claim; files picked by
 *                                  |  the read-balance window (inset below)
 *                                  v
 *         +-------------------------------------------------+
 *         |             SHARED CHUNK QUEUE                  |
 *         |             256 chunks        ~  256 MiB        |
 *         +-------------------------------------------------+
 *                                  |
 *                                  |  28 decode workers
 *                                  |  decode; skip predicate (rfs, ply, wld,
 *                                  |  piece count) keeps ~3% at production;
 *                                  |  zobrist keys update the HLL counters
 *                                  |  inline; kept entries fill a 16k-entry
 *                                  |  scratch (3.5 MiB per worker)
 *                                  v
 *         +-------------------------------------------------+
 *         |              SHUFFLE WINDOW POOL                |
 *         |    12 windows x 2^20 entries  ~  2.6 GiB        |
 *         |                                                 |
 *         |     workers --append(16k bulk, one short mutex) |
 *         |                    |                            |
 *         |                    v                            |
 *         |              [ OPEN WINDOW ]                    |
 *         |                    |  seal at 2^20 entries      |
 *         |                    v                            |
 *         |               [ SHUFFLING ]                     |
 *         |          (sealing worker, outside               |
 *         |           the pool lock)                        |
 *         |                    |                            |
 *         |                    v                            |
 *         |            [ READY (FIFO) queue ]               |
 *         |                                                 |
 *         |     taken by the builders; drained windows      |
 *         |     are returned to the pool and reused         |
 *         +-------------------------------------------------+
 *                                  |
 *                                  |  4 feature builders
 *                                  |  each holds one window, slices batch_size
 *                                  |  (131072) entries per batch, extracts
 *                                  |  features into a SparseBatch (~304 MB)
 *                                  v
 *         +-------------------------------------------------+
 *         |             FINISHED BATCH DEQUE                |
 *         |             8 batches          ~  2.4 GiB       |
 *         +-------------------------------------------------+
 *                                  |
 *                                  |  1 python consumer thread
 *                                  |  fetch_next_sparse_batch -> get_tensors
 *                                  |  -> pinned host memory
 *                                  v
 *         +-------------------------------------------------+
 *         |             PYTHON PREFETCH QUEUE               |
 *         |             2 items (data_loader_queue_size)    |
 *         |                          ~  0.6 GiB             |
 *         +-------------------------------------------------+
 *                                  |
 *                                  v  async H2D
 *                              [ GPU ]
 *
 *       read balance (how the 14 readers pick files):
 *
 *       W = 400 MiB sliding window of recent read bytes per file
 *       (exponentially weighted, kernel exp(-bytes/W)); each file's
 *       target share is its size / total size:
 *
 *         file A (20 GiB)    target |||||||    recent |||      -> deficit, PICK
 *         file B ( 3 GiB)    target |          recent |||      -> satisfied
 *         file C ( 1 GiB)    read in flight                    -> never picked
 *
 *       pick   = idle file with the largest deficit (target - recent share);
 *       claim  = exclusive per file: read ONE ~1 MiB chunk, record its
 *                bytes in the window, release the claim, re-pick
 *       effect = reads spread size-proportionally over all files; a slow
 *                or stuck OST drops out of the rotation automatically and
 *                costs at most one reader thread, never blocking the rest
 *
 *       DDP: every rank seeks to its rank-th chunk and then reads every
 *       world_size-th chunk of each file (chunk-level partitioning).
 */

#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdlib>
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
#include <condition_variable>

#include "rng.h"
#include "thread_safe_types.h"
#include "binpack.h"
#include "training_data_entry.h"
#include "unique_counter.h"
#include "compressed_hash.h"

#include "../training_data_loader_structs.h"


namespace binpack
{
    using namespace std::literals;

    // Shared shuffle window pool.
    //
    // Decode workers cooperatively fill ONE open window with filtered
    // entries; when it reaches capacity it is sealed, shuffled (by the
    // sealing worker, outside the pool lock) and handed to consumers via a
    // FIFO ready queue. Consumers return drained windows to the free list
    // for reuse.
    //
    // This decouples the shuffle window size (which sets the mixing quality
    // of every batch) from the number of decode workers: instead of every
    // worker holding its own full window, only a small pool of windows
    // exists (num_consumer_threads + 4 slots), cutting pipeline host memory
    // by roughly the worker count at unchanged per-batch mixing.
    class ShuffleWindowPool
    {
    public:
        struct Window
        {
            std::vector<TrainingDataEntry> entries;
        };

        ShuffleWindowPool(std::size_t capacity, std::size_t maxWindows) :
            m_capacity(capacity),
            m_maxWindows(maxWindows)
        {
        }

        ShuffleWindowPool(const ShuffleWindowPool&) = delete;
        ShuffleWindowPool& operator=(const ShuffleWindowPool&) = delete;

        // Append entries from scratch (which is consumed). Returns false if
        // the pool was stopped (the caller should abort). A single append
        // may seal multiple windows when the scratch is larger than the
        // capacity.
        bool append(std::vector<TrainingDataEntry>& scratch)
        {
            std::size_t offset = 0;
            while (offset < scratch.size())
            {
                Window* sealed = nullptr;
                {
                    std::unique_lock lock(m_mutex);
                    if (!acquireOpenLocked(lock))
                    {
                        return false;
                    }
                    const std::size_t space = m_capacity - m_open->entries.size();
                    const std::size_t n = std::min(space, scratch.size() - offset);
                    m_open->entries.insert(
                        m_open->entries.end(),
                        std::make_move_iterator(scratch.begin() + offset),
                        std::make_move_iterator(scratch.begin() + offset + n));
                    offset += n;
                    if (m_open->entries.size() >= m_capacity)
                    {
                        sealed = m_open;
                        m_open = nullptr;
                    }
                }

                // Shuffle and deliver outside the pool lock so that other
                // workers can keep filling the next window meanwhile.
                if (sealed != nullptr)
                {
                    auto& prng = rng::get_thread_local_rng();
                    std::shuffle(sealed->entries.begin(), sealed->entries.end(), prng);
                    {
                        std::lock_guard lock(m_mutex);
                        m_ready.push_back(sealed);
                    }
                    m_readyAvailable.notify_one();
                }
            }
            scratch.clear();
            return true;
        }

        // Take the next sealed window for consumption. Returns nullptr once
        // the producers are finished and the ready queue has drained.
        Window* take()
        {
            std::unique_lock lock(m_mutex);
            m_readyAvailable.wait(lock, [this]() {
                return !m_ready.empty() || m_producersDone || m_stopped;
            });
            if (m_ready.empty())
            {
                return nullptr;
            }
            Window* w = m_ready.front();
            m_ready.pop_front();
            return w;
        }

        // Return a fully drained window to the free list.
        void returnWindow(Window* w)
        {
            {
                std::lock_guard lock(m_mutex);
                w->entries.clear();
                m_free.push_back(w);
            }
            m_slotAvailable.notify_one();
        }

        // Called by the last producer: seal the partial open window (if
        // any), hand it to consumers, and mark the pool as finished.
        void finalize()
        {
            Window* sealed = nullptr;
            {
                std::lock_guard lock(m_mutex);
                if (m_open != nullptr && !m_open->entries.empty())
                {
                    sealed = m_open;
                    m_open = nullptr;
                }
            }
            if (sealed != nullptr)
            {
                auto& prng = rng::get_thread_local_rng();
                std::shuffle(sealed->entries.begin(), sealed->entries.end(), prng);
                {
                    std::lock_guard lock(m_mutex);
                    m_ready.push_back(sealed);
                }
            }
            {
                std::lock_guard lock(m_mutex);
                m_producersDone = true;
            }
            m_readyAvailable.notify_all();
        }

        // Wake all blocked producers and consumers (shutdown).
        void signal_stop()
        {
            {
                std::lock_guard lock(m_mutex);
                m_stopped = true;
            }
            m_slotAvailable.notify_all();
            m_readyAvailable.notify_all();
        }


    private:
        // Must hold m_mutex. Ensures m_open refers to an empty window with
        // capacity reserved, waiting for a free slot if the pool is
        // exhausted (backpressure). Returns false on stop. Co-waiting
        // workers share the window opened by whichever of them wakes
        // first, so m_open must be re-checked after every wait.
        bool acquireOpenLocked(std::unique_lock<std::mutex>& lock)
        {
            while (m_open == nullptr)
            {
                if (!m_free.empty())
                {
                    m_open = m_free.back();
                    m_free.pop_back();
                    m_open->entries.clear();
                    m_slotAvailable.notify_all();
                    return true;
                }
                if (m_windows.size() < m_maxWindows)
                {
                    m_windows.push_back(std::make_unique<Window>());
                    m_open = m_windows.back().get();
                    m_open->entries.reserve(m_capacity);
                    m_slotAvailable.notify_all();
                    return true;
                }
                m_slotAvailable.wait(lock, [this]() {
                    return m_open != nullptr || !m_free.empty() || m_stopped;
                });
                if (m_stopped)
                {
                    return false;
                }
            }
            return true;
        }

        std::mutex m_mutex;
        std::condition_variable m_slotAvailable;  // a window became free (producers)
        std::condition_variable m_readyAvailable; // a window became ready (consumers)

        std::vector<std::unique_ptr<Window>> m_windows; // owns all windows
        std::vector<Window*> m_free;                    // drained, reusable
        std::deque<Window*> m_ready;                    // sealed + shuffled, FIFO
        Window* m_open = nullptr;                       // current fill target
        std::size_t m_capacity;
        std::size_t m_maxWindows;
        bool m_producersDone = false;
        bool m_stopped = false;
    };

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
            nnue::UniquePositionCounter* counter = nullptr,
            int rank = 0,
            int world_size = 1,
            DataloaderIOConfig io_config = {},
            int num_consumer_threads = 4
        ) :
            m_concurrency(concurrency),
            m_numRunningWorkers(concurrency),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate)),
            m_counter(counter),
            m_rank(rank),
            m_world_size(world_size)
        {
            std::vector<std::uint64_t> sizes;
            for (const auto& path : paths)
            {
                auto& file = m_inputFiles.emplace_back(path, om | std::ios_base::in);

                if (!file.hasNextChunk()) [[unlikely]]
                {
                     throw std::runtime_error("Empty or corrupted file: " + path);
                }

                sizes.emplace_back(file.sizeBytes());
            }

            for (size_t i = 0; i < m_inputFiles.size(); ++i)
            {
                m_fileMutexes.push_back(std::make_unique<std::timed_mutex>());
            }

            {
                const long long requested = io_config.shuffle_buffer_entries;
                m_threadBufferSize =
                    requested <= 0
                        ? static_cast<std::size_t>(threadBufferSize)
                        : static_cast<std::size_t>(
                              std::clamp<long long>(requested, 1024, 1ll << 24));
            }
            // Shared shuffle windows: capacity = the mixing window (same
            // per-batch mixing as the old per-worker buffers of this size),
            // count = consumers + slack for the open window, in-transit
            // shuffles (a 2^20-entry shuffle takes ~0.1s, so at high
            // consumption several windows are permanently in transit) and
            // the ready backlog.
            m_pool = std::make_unique<ShuffleWindowPool>(
                m_threadBufferSize,
                static_cast<std::size_t>(std::max(1, num_consumer_threads)) + 8);

            // --- Read balancing init ---
            // Reads are balanced on a sliding window of recently read bytes:
            // a file is picked when it is furthest below its size-proportional
            // share of the window. A file whose reads are slow (e.g. its OST is
            // degraded) automatically stops contributing without blocking the
            // other files, because a file with a read in flight is never picked.
            {
                std::uint64_t totalSize = 0;
                for (const auto s : sizes)
                    totalSize += s;

                m_balance.targetShare.resize(sizes.size());
                for (size_t i = 0; i < sizes.size(); ++i)
                    m_balance.targetShare[i] =
                        totalSize > 0
                            ? static_cast<double>(sizes[i]) / static_cast<double>(totalSize)
                            : 1.0 / static_cast<double>(sizes.size());
                m_balance.recentBytes.assign(sizes.size(), 0);
                m_balance.claimed.assign(sizes.size(), 0);

                const int windowMb =
                    io_config.balance_window_mb > 0 ? io_config.balance_window_mb
                                                    : kDefaultBalanceWindowMb;
                m_balance.windowLimitBytes =
                    std::max<std::uint64_t>(static_cast<std::uint64_t>(windowMb) * MiB,
                                            16 * MiB);
            }

            m_ioStats = std::make_unique<FileIoStats[]>(sizes.size());

            m_simSlowDelayMs = parseSimSlowEnv(sizes.size());
            if (!m_simSlowDelayMs.empty())
            {
                std::cerr << "[Info] NNUE_LOADER_SIM_SLOW active:";
                for (size_t i = 0; i < m_simSlowDelayMs.size(); ++i)
                {
                    if (m_simSlowDelayMs[i])
                        std::cerr << " file " << i << " ("
                                  << m_inputFiles[i].path() << ") +" << m_simSlowDelayMs[i] << "ms";
                }
                std::cerr << std::endl;
            }

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

                    // Pick the idle (not exhausted, no read in flight) file that
                    // is furthest below its size-proportional share of the
                    // balance window. The claim happens atomically with the
                    // pick. Returns kInvalidFile if every file is busy.
                    std::size_t fileId = claimBestFile();
                    if (fileId == kInvalidFile)
                    {
                        waitUntilFileAvailable();
                        continue;
                    }

                    // Safety net so the claim is released even if an exception
                    // escapes the read. Normally released explicitly below.
                    struct ClaimGuard
                    {
                        CompressedTrainingDataEntryParallelReader* self;
                        std::size_t id;
                        bool active = true;

                        ~ClaimGuard()
                        {
                            if (active)
                                self->releaseClaim(id);
                        }
                    } claimGuard{this, fileId};

                    std::unique_lock lock(*m_fileMutexes[fileId]);
                    // The claim guarantees this mutex is uncontended; it
                    // serializes access to the file's stream state.

                    auto& inputFile = m_inputFiles[fileId];

                    auto seek_for_ddp_rank = [&](std::size_t rank) -> bool
                    {
                        std::size_t skipped = 0;
                        if (inputFile.skipChunks(rank, &skipped))
                        {
                            // Guard against landing exactly at EOF (possible
                            // when the file has no more than `rank` chunks);
                            // reading there would fail.
                            return inputFile.hasNextChunk();
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

                    // Test hook: simulate a degraded OST for selected files via
                    // NNUE_LOADER_SIM_SLOW="idx:delay_ms[,...]". The delay runs
                    // under the claim and inside the timed section, exactly
                    // like a genuinely slow read.
                    const unsigned simDelayMs =
                        m_simSlowDelayMs.empty() ? 0u : m_simSlowDelayMs[fileId];

                    m_ioStats[fileId].readStartNs.store(steadyNs(), std::memory_order_relaxed);
                    const auto t0 = std::chrono::steady_clock::now();
                    if (simDelayMs)
                        std::this_thread::sleep_for(std::chrono::milliseconds(simDelayMs));
                    std::vector<unsigned char> chunk = inputFile.readNextChunk();
                    const auto readNs = static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count());
                    m_ioStats[fileId].readStartNs.store(0, std::memory_order_relaxed);

                    {
                        auto& st = m_ioStats[fileId];
                        st.chunks.fetch_add(1, std::memory_order_relaxed);
                        st.bytes.fetch_add(chunk.size(), std::memory_order_relaxed);
                        st.nsTotal.fetch_add(readNs, std::memory_order_relaxed);
                        st.lastNs.store(readNs, std::memory_order_relaxed);
                        std::uint64_t prevMax = st.nsMax.load(std::memory_order_relaxed);
                        while (readNs > prevMax
                               && !st.nsMax.compare_exchange_weak(
                                      prevMax, readNs, std::memory_order_relaxed,
                                      std::memory_order_relaxed))
                        {
                        }
                    }

                    if (m_world_size > 1)
                    {
                        m_ddp_chunks_to_skip_after_read[fileId] = static_cast<std::size_t>(m_world_size - 1);
                    }

                    lock.unlock();

                    // Record the read in the balance window before releasing
                    // the claim so concurrent pickers see fresh priorities.
                    recordRead(fileId, static_cast<std::uint32_t>(chunk.size()));

                    releaseClaim(fileId);
                    claimGuard.active = false;

                    if (readNs > kSlowReadWarnNs)
                        warnSlowRead(fileId, readNs);

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
                ChunkReader m_chunkReader{};
                // Small decode scratch; filtered entries are appended to
                // the shared shuffle window pool in bulk.
                std::vector<TrainingDataEntry> scratch;
                scratch.reserve(kWorkerScratchEntries);

                constexpr std::size_t keyFlushThreshold = 4096;
                std::vector<std::uint64_t> keyBuffer;
                std::uint64_t preskip_count = 0;
                if (m_counter) keyBuffer.reserve(keyFlushThreshold);

                auto flushAll = [&]() {
                    if (m_counter) {
                        if (!keyBuffer.empty()) {
                            m_counter->addBatch(std::move(keyBuffer));
                            keyBuffer.clear();
                            keyBuffer.reserve(keyFlushThreshold);
                        }
                        if (preskip_count > 0) {
                            m_counter->addPreskip(preskip_count);
                            preskip_count = 0;
                        }
                    }
                };

                // Finalizes producer accounting on every exit path (also on
                // exceptions), so consumers can never hang on a lost
                // producer. The last worker to exit seals the final partial
                // window and marks the pool as finished.
                struct WorkerExit
                {
                    CompressedTrainingDataEntryParallelReader* self;
                    ~WorkerExit()
                    {
                        if (self->m_numRunningWorkers.fetch_sub(1) == 1)
                        {
                            self->m_pool->finalize();
                        }
                    }
                } workerExit{this};

                bool isEnd = fetchNextChunkFromSharedQueue(m_chunkReader, m_chunk);

                while(!isEnd && !m_stopFlag.load())
                {
                    const auto e = m_chunkReader.next(m_chunk);
                    if (m_counter) ++preskip_count;

                    if (!m_chunkReader.hasNext(m_chunk))
                    {
                        isEnd = fetchNextChunkFromSharedQueue(m_chunkReader, m_chunk);
                    }

                    if (!m_skipPredicate || !m_skipPredicate(e))
                    {
                        scratch.emplace_back(e);
                        if (m_counter)
                        {
                            keyBuffer.push_back(nnue::hash::hash(e.pos));
                            if (keyBuffer.size() >= keyFlushThreshold)
                                flushAll();
                        }
                        if (scratch.size() >= kWorkerScratchEntries)
                        {
                            if (!m_pool->append(scratch))
                            {
                                flushAll();
                                return; // stopped
                            }
                        }
                    }
                }

                if (!scratch.empty() && !m_stopFlag.load())
                {
                    m_pool->append(scratch);
                }

                flushAll();
            };

            for (int i = 0; i < concurrency; ++i)
            {
                m_workers.emplace_back(worker);
            }
        }

        [[nodiscard]] std::optional<TrainingDataEntry> next()
        {
            LocalBuffer& local = m_bufferRegistry.get();
            if (local.window != nullptr && local.offset < local.window->entries.size()) [[likely]]
            {
                return std::move(local.window->entries[local.offset++]);
            }

            if (local.window != nullptr)
            {
                m_pool->returnWindow(local.window);
                local.window = nullptr;
            }

            local.window = m_pool->take();
            if (local.window == nullptr)
            {
                return std::nullopt;
            }
            local.offset = 0;

            return std::move(local.window->entries[local.offset++]);
        }

        int fill(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            LocalBuffer& local = m_bufferRegistry.get();
            std::size_t total_filled = 0;

            while (total_filled < n) {
                if (local.window == nullptr || local.offset >= local.window->entries.size()) [[unlikely]]
                {
                    if (local.window != nullptr)
                    {
                        m_pool->returnWindow(local.window);
                        local.window = nullptr;
                    }
                    local.window = m_pool->take();
                    if (local.window == nullptr) break; // Pool and workers exhausted
                    local.offset = 0;
                }

                const std::size_t available = local.window->entries.size() - local.offset;
                const std::size_t to_copy = std::min(n - total_filled, available);

                vec.insert(
                    vec.end(),
                    std::make_move_iterator(local.window->entries.begin() + local.offset),
                    std::make_move_iterator(local.window->entries.begin() + local.offset + to_copy)
                );

                local.offset += to_copy;
                total_filled += to_copy;
            }
            return static_cast<int>(total_filled);
        }

        // Fill per-file I/O statistics for read-balancing observability.
        // If out is null or max_files is 0, only returns the file count.
        // Race-free; may be called while the reader is running.
        std::size_t get_io_stats(DataloaderFileStats* out, std::size_t max_files)
        {
            const std::size_t n = m_inputFiles.size();
            if (out == nullptr || max_files == 0)
                return n;

            const std::size_t m = std::min(n, max_files);
            for (std::size_t i = 0; i < m; ++i)
            {
                auto& st = m_ioStats[i];
                auto& o = out[i];
                o.chunks_read = st.chunks.load(std::memory_order_relaxed);
                o.bytes_read = st.bytes.load(std::memory_order_relaxed);
                o.read_ns_total = st.nsTotal.load(std::memory_order_relaxed);
                o.read_ns_max = st.nsMax.load(std::memory_order_relaxed);
                o.last_read_ns = st.lastNs.load(std::memory_order_relaxed);
                const std::uint64_t start = st.readStartNs.load(std::memory_order_relaxed);
                o.read_started_ms_ago =
                    start ? static_cast<std::int64_t>((steadyNs() - start) / 1000000) : 0;
                o.exhausted = m_fileExhausted[i].load(std::memory_order_relaxed) ? 1 : 0;
            }
            {
                std::lock_guard lock(m_balance.mutex);
                for (std::size_t i = 0; i < m; ++i)
                {
                    out[i].window_bytes =
                        static_cast<std::uint64_t>(m_balance.recentBytes[i]);
                    out[i].claimed = m_balance.claimed[i] ? 1 : 0;
                }
            }
            return n;
        }

        ~CompressedTrainingDataEntryParallelReader()
        {
            m_stopFlag.store(true);
            m_sharedChunkQueue.signal_stop();
            m_pool->signal_stop();
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

        // Capacity of one shared shuffle window, in entries (post-filter).
        // The static value above is the default. A window of this many
        // entries defines the mixing window every batch is drawn from; only
        // a small pool of them exists (consumers + 4), instead of one per
        // decode worker as before.
        std::size_t m_threadBufferSize;

        // Per-worker decode scratch that is appended to the shared shuffle
        // window pool in bulk (16k entries = 3.5 MiB per worker).
        static constexpr std::size_t kWorkerScratchEntries = 1 << 14;

        // Shared shuffle window pool (owns all windows).
        std::unique_ptr<ShuffleWindowPool> m_pool;

        std::atomic_bool m_stopFlag;
        std::vector<std::thread> m_workers;

        // Per File Lock
        std::vector<std::unique_ptr<std::timed_mutex>> m_fileMutexes;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
        nnue::UniquePositionCounter* m_counter = nullptr;

        // ---- Read balancing: sliding window + per-file claims ----
        // Reads are picked by argmax deficit (size-proportional share of the
        // window minus recent bytes) among files that are idle. A file with a
        // read in flight is never picked, so a slow or stuck file costs at
        // most one reader thread and never blocks reads of the other files.
        static constexpr std::size_t kInvalidFile = static_cast<std::size_t>(-1);
        static constexpr int kDefaultBalanceWindowMb = 400;
        static constexpr std::uint64_t kSlowReadWarnNs =
            std::uint64_t{5} * 1000 * 1000 * 1000;
        static constexpr int64_t kSlowReadWarnCooldownSeconds = 300;
        static constexpr std::chrono::milliseconds kAvailableWaitTimeout{10};

        struct ReadBalance
        {
            std::mutex mutex;
            std::condition_variable anyAvailable;
            // Exponentially weighted window over recently read bytes per file
            // (kernel exp(-bytes/W)); decays all files on every record. This
            // is a continuously moving window and avoids the lumpy
            // replenishment of FIFO event eviction, which biases small files.
            std::vector<double> recentBytes;
            double totalBytes = 0.0;
            std::uint64_t windowLimitBytes = 0;
            std::vector<double> targetShare;
            std::vector<char> claimed; // guarded by mutex
        } m_balance;

        struct FileIoStats
        {
            std::atomic<std::uint64_t> chunks{0};
            std::atomic<std::uint64_t> bytes{0};
            std::atomic<std::uint64_t> nsTotal{0};
            std::atomic<std::uint64_t> nsMax{0};
            std::atomic<std::uint64_t> lastNs{0};
            std::atomic<std::uint64_t> readStartNs{0}; // steady ns of in-flight read; 0 = idle
        };
        std::unique_ptr<FileIoStats[]> m_ioStats;

        // Test hook (per file, milliseconds, 0 = disabled).
        std::vector<unsigned> m_simSlowDelayMs;
        std::atomic<int64_t> m_lastSlowReadWarnSec{-kSlowReadWarnCooldownSeconds};

        // DDP support
        int m_rank;
        int m_world_size;
        std::vector<std::uint8_t> m_files_seeked_for_ddp;  // Track which files have been seeked for DDP
        std::vector<std::size_t> m_ddp_chunks_to_skip_after_read;

        // thread local consumer buffers: each consumer thread drains at
        // most one pool window at a time.
        struct alignas(128) LocalBuffer {
            ShuffleWindowPool::Window* window = nullptr;
            size_t offset = 0;
        };

        thread_safe_types::ThreadLocalRegistry<LocalBuffer> m_bufferRegistry;

        bool fetchNextChunkFromSharedQueue(ChunkReader& m_chunkReader, std::vector<unsigned char>& m_chunk)
        {
            if (!m_chunkReader.hasNext(m_chunk))
            {
                if (m_stopFlag.load())
                {
                    return true;
                }

                if (m_readersFinished.load() && m_sharedChunkQueue.is_empty())
                {
                    return true;
                }

                // Note: the predicate runs while take() holds the queue's
                // mutex; calling m_sharedChunkQueue.is_empty() here would
                // re-lock it and deadlock the worker (pre-existing bug,
                // triggered when readers finish while workers race to drain
                // the last chunks). take() only returns false after the
                // queue has drained (it re-checks m_ringCount after the
                // wait), so "readers finished" alone is a safe stop
                // condition and no buffered data is lost.
                bool success = m_sharedChunkQueue.take(
                    m_chunk,
                    [this]() { return m_stopFlag.load() || m_readersFinished.load(); }
                );

                if (success)
                {
                    m_chunkReader = ChunkReader{};
                    return false;
                }
                return true;
            }

            return false;
        }

        static std::uint64_t steadyNs()
        {
            return static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
        }

        std::size_t claimBestFile()
        {
            std::lock_guard lock(m_balance.mutex);
            std::size_t best = kInvalidFile;
            double bestPriority = 0.0;
            const double windowBytes = m_balance.totalBytes;
            for (std::size_t i = 0; i < m_inputFiles.size(); ++i)
            {
                if (m_fileExhausted[i].load(std::memory_order_relaxed) || m_balance.claimed[i])
                    continue;
                const double priority =
                    m_balance.targetShare[i] * windowBytes
                    - static_cast<double>(m_balance.recentBytes[i]);
                if (best == kInvalidFile || priority > bestPriority)
                {
                    best = i;
                    bestPriority = priority;
                }
            }
            if (best != kInvalidFile)
                m_balance.claimed[best] = 1;
            return best;
        }

        void releaseClaim(std::size_t fileId)
        {
            {
                std::lock_guard lock(m_balance.mutex);
                m_balance.claimed[fileId] = 0;
            }
            m_balance.anyAvailable.notify_all();
        }

        bool anyFileAvailableLocked()
        {
            for (std::size_t i = 0; i < m_inputFiles.size(); ++i)
            {
                if (!m_fileExhausted[i].load(std::memory_order_relaxed) && !m_balance.claimed[i])
                    return true;
            }
            return false;
        }

        void waitUntilFileAvailable()
        {
            std::unique_lock lock(m_balance.mutex);
            m_balance.anyAvailable.wait_for(lock, kAvailableWaitTimeout, [this]() {
                return m_stopFlag.load() || anyFileAvailableLocked();
            });
        }

        void recordRead(std::size_t fileId, std::uint32_t bytes)
        {
            std::lock_guard lock(m_balance.mutex);
            // Exponential decay of the whole window by the new sample, then
            // credit the reader. The window only moves as data is read.
            const double factor =
                std::exp(-static_cast<double>(bytes)
                         / static_cast<double>(m_balance.windowLimitBytes));
            for (auto& r : m_balance.recentBytes)
                r *= factor;
            m_balance.recentBytes[fileId] += static_cast<double>(bytes);
            m_balance.totalBytes = m_balance.totalBytes * factor
                                 + static_cast<double>(bytes);
        }

        void warnSlowRead(std::size_t fileId, std::uint64_t readNs)
        {
            const auto now =
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            int64_t last = m_lastSlowReadWarnSec.load(std::memory_order_relaxed);
            if (now - last >= kSlowReadWarnCooldownSeconds
                && m_lastSlowReadWarnSec.compare_exchange_strong(
                       last, now, std::memory_order_relaxed, std::memory_order_relaxed))
            {
                std::cerr << "[Warning] Dataloader read from "
                          << m_inputFiles[fileId].path() << " took "
                          << readNs / 1000000 << " ms. Storage degraded?\n";
            }
        }

        // Test hook: NNUE_LOADER_SIM_SLOW="idx:delay_ms[,idx:delay_ms...]"
        // adds an artificial per-chunk delay before reading the given files,
        // simulating a degraded OST. Parsed once per reader.
        static std::vector<unsigned> parseSimSlowEnv(std::size_t numFiles)
        {
            std::vector<unsigned> delays;
            const char* env = std::getenv("NNUE_LOADER_SIM_SLOW");
            if (env == nullptr || *env == '\0')
                return delays;

            try
            {
                const std::string spec(env);
                std::vector<std::pair<std::size_t, unsigned>> parsed;
                std::size_t start = 0;
                for (;;)
                {
                    const std::size_t comma = spec.find(',', start);
                    const std::string token =
                        spec.substr(start, comma == std::string::npos
                                               ? std::string::npos
                                               : comma - start);
                    const std::size_t colon = token.find(':');
                    if (colon == std::string::npos)
                        throw std::invalid_argument("expected idx:delay_ms");
                    const std::size_t idx = std::stoull(token.substr(0, colon));
                    const unsigned ms =
                        static_cast<unsigned>(std::stoul(token.substr(colon + 1)));
                    if (idx >= numFiles)
                        throw std::out_of_range("file index out of range");
                    parsed.emplace_back(idx, ms);
                    if (comma == std::string::npos)
                        break;
                    start = comma + 1;
                }
                delays.assign(numFiles, 0);
                for (const auto& [idx, ms] : parsed)
                    delays[idx] = ms;
            }
            catch (...)
            {
                std::cerr << "[Warning] Ignoring invalid NNUE_LOADER_SIM_SLOW value: "
                          << env << std::endl;
                delays.clear();
            }
            return delays;
        }

        // Shared Raw Chunk Queue
        thread_safe_types::ThreadSafeRingBuffer<FileChunk, sharedChunkQueueCapacity> m_sharedChunkQueue;

        std::unique_ptr<std::atomic_bool[]> m_fileExhausted;
        std::vector<std::thread> m_readerThreads;
        std::atomic_bool m_readersFinished;
        std::atomic_int m_numRunningReaders;
    };

}
