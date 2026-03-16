#pragma once

#include <array>
#include <stdexcept>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <concepts>

namespace thread_safe_types
{
    template <typename T, std::size_t Capacity>
    class ThreadSafeRingBuffer {
        static_assert(Capacity > 0, "Capacity must be greater than 0.");

    public:
        ThreadSafeRingBuffer() {
            m_ringBuffer.resize(Capacity);
        }

        /**
        * @brief Swaps the provided item into the ring buffer.
        * @return true if successful, false if the stopCondition triggered.
        */
        template <std::predicate<> Pred>
        bool put(T& incomingItem, Pred stopCondition) {
            std::unique_lock lock(m_ringMutex);

            m_ringNotFull.wait(lock, [this, &stopCondition]() {
                return m_ringCount < Capacity || stopCondition();
            });

            // Check if we woke up because of stopCondition and the buffer is still full
            if (stopCondition() && m_ringCount == Capacity) {
                return false;
            }

            m_ringBuffer[m_ringTail].swap(incomingItem);
            m_ringTail = (m_ringTail + 1) % Capacity;
            m_ringCount++;

            lock.unlock();
            m_ringNotEmpty.notify_one();
            return true;
        }

        /**
        * @brief Swaps an item from the ring buffer into the provided item.
        * @return true if successful, false if the stopCondition triggered or empty.
        */

        template <std::predicate<> Pred>
        bool take(T& outgoingItem, Pred stopCondition) {
            std::unique_lock lock(m_ringMutex);

            m_ringNotEmpty.wait(lock, [this, &stopCondition]() {
                return m_ringCount > 0 || stopCondition();
            });

            if (m_ringCount == 0) {
                return false;
            }

            m_ringBuffer[m_ringHead].swap(outgoingItem);
            m_ringHead = (m_ringHead + 1) % Capacity;
            m_ringCount--;

            lock.unlock();
            m_ringNotFull.notify_one();
            return true;
        }

        void signal_stop(bool signalProducers=true) {
            m_ringNotEmpty.notify_all();
            if(signalProducers)
                m_ringNotFull.notify_all();
        }

        void reserve_internal(size_t size) {
            for (auto& slot : m_ringBuffer) {
                slot.reserve(size);
            }
        }

    private:
        std::vector<T> m_ringBuffer;
        std::mutex m_ringMutex;
        std::condition_variable m_ringNotEmpty;
        std::condition_variable m_ringNotFull;

        alignas(64) size_t m_ringHead = 0;
        alignas(64) size_t m_ringTail = 0;
        alignas(64) size_t m_ringCount = 0;
    };

    template <typename T>
    class ThreadSafeRingBuffer<T, 1> {
    public:
        ThreadSafeRingBuffer() {
            m_slot.resize(1);
        }

        template <class Pred>
        requires std::predicate<Pred>
        bool put(T& incomingItem, Pred stopCondition) {
            std::unique_lock lock(m_ringMutex);

            m_ringNotFull.wait(lock, [this, &stopCondition]() {
                return !m_isFull || stopCondition();
            });

            if (stopCondition() && m_isFull) return false;

            m_slot[0].swap(incomingItem);
            m_isFull = true;

            lock.unlock();
            m_ringNotEmpty.notify_one();
            return true;
        }

        template <class Pred>
        requires std::predicate<Pred>
        bool take(T& outgoingItem, Pred stopCondition) {
            std::unique_lock lock(m_ringMutex);

            m_ringNotEmpty.wait(lock, [this, &stopCondition]() {
                return m_isFull || stopCondition();
            });

            if (!m_isFull) return false;

            m_slot[0].swap(outgoingItem);
            m_isFull = false;

            lock.unlock();
            m_ringNotFull.notify_one();
            return true;
        }

        void signal_stop(bool signalProducers=true) {
            m_ringNotEmpty.notify_all();
            if(signalProducers)
                m_ringNotFull.notify_all();
        }


        void reserve_internal(size_t size) {
            m_slot.reserve(size);
        }

    private:
        std::vector<T> m_slot; // Using vector so .swap() works as expected
        bool m_isFull = false;
        std::mutex m_ringMutex;
        std::condition_variable m_ringNotEmpty;
        std::condition_variable m_ringNotFull;
    };

    class ThreadIdManager {
    public:
        static uint32_t acquire() {
            std::lock_guard<std::mutex> lock(getMutex());
            auto& freeIds = getFreeIds();

            if (!freeIds.empty()) {
                uint32_t id = freeIds.back();
                freeIds.pop_back();
                return id;
            }

            return getMaxId().fetch_add(1, std::memory_order_relaxed);
        }

        static void release(uint32_t id) {
            std::lock_guard<std::mutex> lock(getMutex());
            getFreeIds().push_back(id);
        }

    private:
        static std::mutex& getMutex() {
            // leaked on purpose to prevent UB at termination
            static std::mutex* m = new std::mutex();
            return *m;
        }

        static std::vector<uint32_t>& getFreeIds() {
            // leaked on purpose to prevent UB at termination
            static std::vector<uint32_t>* ids = new std::vector<uint32_t>();
            return *ids;
        }

        static std::atomic<uint32_t>& getMaxId() {
            static std::atomic<uint32_t> maxId{0};
            return maxId;
        }
    };

    struct ThreadLocalIndex {
        const uint32_t id;

        ThreadLocalIndex() : id(ThreadIdManager::acquire()) {}

        ~ThreadLocalIndex() {
            ThreadIdManager::release(id);
        }

        static uint32_t get() {
            static thread_local ThreadLocalIndex current;
            return current.id;
        }
    };


    template <typename T, uint32_t ChunkSize = 512, uint32_t MaxChunks = 8>
    class ThreadLocalRegistry {
    private:
        struct Chunk {
            std::array<T, ChunkSize> buffers;
        };

        std::array<std::atomic<Chunk*>, MaxChunks> m_chunks{};

    public:
        // Preallocate a specific number of chunks on construction
        explicit ThreadLocalRegistry(uint32_t preallocated_chunks = 2) {
            if (preallocated_chunks > MaxChunks) {
                preallocated_chunks = MaxChunks;
            }
            for (uint32_t i = 0; i < preallocated_chunks; ++i) {
                m_chunks[i].store(new Chunk(), std::memory_order_relaxed);
            }
        }

        ~ThreadLocalRegistry() {
            for (auto& atomic_chunk : m_chunks) {
                delete atomic_chunk.load(std::memory_order_relaxed);
            }
        }

        // Delete copy/move semantics to guarantee absolute pointer stability
        ThreadLocalRegistry(const ThreadLocalRegistry&) = delete;
        ThreadLocalRegistry& operator=(const ThreadLocalRegistry&) = delete;

        T& get(uint32_t thread_idx = ThreadLocalIndex::get()) {
            const uint32_t chunk_idx = thread_idx / ChunkSize;
            const uint32_t local_idx = thread_idx % ChunkSize;

            if (chunk_idx >= MaxChunks) [[unlikely]] {
                throw std::out_of_range("Thread index exceeds registry capacity.");
            }

            // HOT PATH: Lock-free read
            Chunk* chunk = m_chunks[chunk_idx].load(std::memory_order_acquire);

            if (chunk != nullptr) [[likely]] {
                return chunk->buffers[local_idx];
            }

            // COLD PATH: Allocate new chunk via CAS
            Chunk* new_chunk = new Chunk();
            Chunk* expected = nullptr;

            if (m_chunks[chunk_idx].compare_exchange_strong(
                    expected, new_chunk,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                return new_chunk->buffers[local_idx];
            } else {
                // Lost the race, clean up and use the winner's chunk
                delete new_chunk;
                return expected->buffers[local_idx];
            }
        }
    };
}