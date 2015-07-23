#include <cassert>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cpu_runtime.h"
#include "thorin_utils.h"
#include "thorin_runtime.h"

#ifdef USE_TBB
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#endif

// helper functions
void thorin_init() { }
void* thorin_malloc(uint32_t size) {
    void* mem = thorin_aligned_malloc(size, 64);
#ifndef NDEBUG
    std::clog << " * malloc(" << size << ") -> " << mem << std::endl;
#endif
    return mem;
}
void thorin_free(void* ptr) {
    thorin_aligned_free(ptr);
}
void thorin_print_total_timing() { }

#ifndef USE_TBB
static std::unordered_map<int, std::thread> thread_pool;
static std::vector<int> free_ids;

// C++11 threads version
void parallel_for(int num_threads, int lower, int upper, void* args, void* fun) {
    // Get number of available hardware threads
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        // hardware_concurrency is implementation defined, may return 0
        num_threads = (num_threads == 0) ? 1 : num_threads;
    }

    void (*fun_ptr) (void*, int, int) = reinterpret_cast<void (*) (void*, int, int)>(fun);
    const int linear = (upper - lower) / num_threads;

    // Create a pool of threads to execute the task
    std::vector<std::thread> pool(num_threads);

    for (int i = 0, a = lower, b = lower + linear; i < num_threads - 1; a = b, b += linear, i++) {
        pool[i] = std::thread([=]() {
            fun_ptr(args, a, b);
        });
    }

    pool[num_threads - 1] = std::thread([=]() {
        fun_ptr(args, lower + (num_threads - 1) * linear, upper);
    });

    // Wait for all the threads to finish
    for (int i = 0; i < num_threads; i++)
        pool[i].join();
}
int parallel_spawn(void* args, void* fun) {
    int (*fun_ptr) (void*) = reinterpret_cast<int (*) (void*)>(fun);

    int id;
    if (free_ids.size()) {
        id = free_ids.back();
        free_ids.pop_back();    
    } else {
        id = thread_pool.size();
    }

    auto spawned = std::make_pair(id, std::thread([=](){ fun_ptr(args); }));
    thread_pool.emplace(std::move(spawned));
    return id;
}
void parallel_sync(int id) {
    auto thread = thread_pool.find(id);
    if (thread != thread_pool.end()) {
        thread->second.join();
        free_ids.push_back(thread->first);
        thread_pool.erase(thread);
    } else {
        assert(0 && "Trying to synchronize on invalid thread id");
    }
}
#else
// TBB version
void parallel_for(int num_threads, int lower, int upper, void* args, void* fun) {
    tbb::task_scheduler_init init((num_threads == 0) ? tbb::task_scheduler_init::automatic : num_threads);
    void (*fun_ptr) (void*, int, int) = reinterpret_cast<void (*) (void*, int, int)>(fun);

    tbb::parallel_for(tbb::blocked_range<int>(lower, upper), [=] (const tbb::blocked_range<int>& range) {
        fun_ptr(args, range.begin(), range.end());
    });
}

static std::unordered_map<int, tbb::task*> task_pool;
static std::vector<int> free_ids;

class RuntimeTask : public tbb::task {
public:
    RuntimeTask(void* args, void* fun)
        : args_(args), fun_(fun)
    {}

    tbb::task* execute() {
        int (*fun_ptr) (void*) = reinterpret_cast<int (*) (void*)>(fun_);
        fun_ptr(args_);
        set_ref_count(1);
        return nullptr;
    }

private:
    void* args_;
    void* fun_;
};

int parallel_spawn(void* args, void* fun) {
    int id;
    if (free_ids.size()) {
        id = free_ids.back();
        free_ids.pop_back();
    } else {
        id = task_pool.size();
    }

    tbb::task* task = new (tbb::task::allocate_root()) RuntimeTask(args, fun);
    tbb::task::spawn(*task);
    task_pool[id] = task;
    return id;
}
void parallel_sync(int id) {
    auto task = task_pool.find(id);
    if (task != task_pool.end()) {
        task->second->wait_for_all();
        tbb::task::destroy(*task->second);
        free_ids.push_back(task->first);
        task_pool.erase(task);
    } else {
        assert(0 && "Trying to synchronize on invalid task id");
    }
}
#endif

