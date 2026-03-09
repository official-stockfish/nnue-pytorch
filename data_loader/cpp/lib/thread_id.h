#pragma once

#include <vector>
#include <mutex>
#include <atomic>
#include <cstdint>

namespace thread_id
{
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
            static std::mutex m;
            return m;
        }

        static std::vector<uint32_t>& getFreeIds() {
            static std::vector<uint32_t> ids;
            return ids;
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
}