#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <array>
#include <iomanip>
#include <iostream>
#include <fstream>
// #include <memory>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <fcntl.h>
// #include <limits>
// #include <bitset>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include <cstdio>
#include <ctime>
#include <string>
#include <sys/resource.h>
#include <sys/time.h>

// #define DEBUG

// #undef __x86_64__
// #undef __aarch64__
// #define __aarch64__

#if defined(__x86_64__)
  #include <crc32intrin.h>
  #include <nmmintrin.h>
  #include <immintrin.h>
  #ifdef __AVX512F__
    using VecType = __m512i;
    using MaskType = __mmask64;
    const int VecSize = 64;
    #define VEC_SET1_EPI8 _mm512_set1_epi8
    #define VEC_LOADU_SI _mm512_loadu_si512
    #define VEC_LOAD_SI _mm512_load_si512
    #define VEC_CMPEQ_EPI8_MASK _mm512_cmpeq_epi8_mask
    #define VEC_MOVEMASK_EPI8 _mm512_movemask_epi8
    #define VEC_STORE_SI(buffer, data) _mm512_store_si512(reinterpret_cast<__m512i*>(buffer), data)
    #define TZCNT _tzcnt_u64
  #else
    using VecType = __m256i;
    using MaskType = __mmask32;
    const int VecSize = 32;
    #define VEC_SET1_EPI8 _mm256_set1_epi8
    #define VEC_LOADU_SI _mm256_loadu_si256
    #define VEC_LOAD_SI _mm256_load_si256
    #define VEC_CMPEQ_EPI8_MASK(data, val) _mm256_movemask_epi8(_mm256_cmpeq_epi8(data, val))
    #define VEC_STORE_SI(buffer, data) _mm256_store_si256(reinterpret_cast<__m256i*>(buffer), data)
    #define TZCNT _tzcnt_u32
  #endif
  #define CRC32_8 _mm_crc32_u8
  #define CRC32_32 _mm_crc32_u32
  #define CRC32_64 _mm_crc32_u64
#elif defined(__aarch64__)
  #include <arm_neon.h>
  #include <arm_acle.h>

  using VecType = uint8x16_t;
  using MaskType = uint16_t;  // NEON lacks a direct equivalent to __mmask32. Using 16-bit mask for 16 8-bit elements.
  const int VecSize = 16;     // Adjusted for 128-bit vector
  #define VEC_SET1_EPI8 vdupq_n_s8

  #define VEC_LOADU_SI(ptr) vld1q_s8(reinterpret_cast<const int8_t*>(ptr))
  #define VEC_LOAD_SI(ptr) vld1q_s8(reinterpret_cast<const int8_t*>(ptr))

  #define VEC_CMPEQ_EPI8_MASK(data, val) ({ \
    uint8x16_t result_vec = vceqq_s8(data, val); \
    uint16_t result_mask = 0; \
    for (int i = 0; i < 16; ++i) { \
        if (result_vec[i] != 0) { \
            result_mask |= (1U << i); \
        } \
    } \
    result_mask; \
  })

  #define VEC_STORE_SI(buffer, data) vst1q_s8(reinterpret_cast<int8_t*>(buffer), data)
  #define TZCNT __builtin_ctz

  #define CRC32_8 __crc32b
  #define CRC32_32 __crc32w
  #define CRC32_64 __crc32d
#else
  #error "Unsupported platform. CRC32 intrinsic not defined."
#endif

#define yay(x) __builtin_expect(x, 1)
#define nay(x) __builtin_expect(x, 0)
#define inline inline __attribute__((always_inline))

// It's a timer.
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    int64_t nanoseconds() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - start_).count();
    }

    double milliseconds() const {
        int64_t nanos = nanoseconds();
        return static_cast<double>(nanos) / 1000000;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// It's a logger.
template <typename... Args> void info(Args const&... args)
{
    std::ios_base::fmtflags f(std::cout.flags());
    ((std::cout << args), ...) << std::endl;
    std::cout.flags(f);
}

#ifdef DEBUG
template <typename... Args> void debug(Args const&... args)
{
    info(args...);
}
#else
//template <typename... Args> void debug(Args const&... args){}
#define debug(...)
#endif

class ThreadPool {
public:
    ThreadPool(size_t num_threads, int pin_offset = 0) {
        for (size_t i = 0; i < num_threads; ++i) {
            threads.emplace_back([=] { this->worker_thread(pin_offset + i); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stopping = true;
        }
        condition.notify_all();
        for (auto &thread : threads) {
            thread.join();
        }
    }

    void enqueue(std::function<void(int)> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.push(std::move(task));
        }
        condition.notify_one();
    }

private:
    std::vector<std::thread> threads;
    std::queue<std::function<void(int)>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stopping = false;

    void worker_thread(int i) {
#if defined(__x86_64__)
        // Set thread affinity to CPU core 'i'.
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);

        int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            info("Error setting thread affinity: ", i, strerror(rc));
        }
#endif
        while (true) {
            std::function<void(int)> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                condition.wait(lock, [this] { return stopping || !tasks.empty(); });
                if (stopping && tasks.empty()) {
                    break;
                }
                task = std::move(tasks.front());
                tasks.pop();
            }
            task(i);
        }
    }
};

class TaskManager {
public:
    TaskManager(ThreadPool& pool) : pool(pool), active_tasks(0) {}

    void enqueue(std::function<void(int)> task) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        ++active_tasks;
        pool.enqueue([this, task] (int i) {
            task(i);
            std::unique_lock<std::mutex> lock(queue_mutex);
            --active_tasks;
            if (active_tasks == 0) {
                condition.notify_all();
            }
        });
    }

    void flush() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        condition.wait(lock, [this] { return active_tasks == 0; });
    }

private:
    ThreadPool& pool;
    std::atomic<int> active_tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
};

// A big 'ol number lookup table.
constexpr int NUM_LOOKUP_TABLE_SIZE = 1<<15;
int num_lookup[NUM_LOOKUP_TABLE_SIZE];

// Computes the key for looking up in the number lookup table.
// Potential value formats where n is newline.
// -99.9n
// -9.9n
// 99.9n
// 9.9n
inline int gen_num_key(const char* data, int newline_index) {
    uint32_t i = *(uint32_t*)(data);
    uint32_t k = CRC32_32(0, i);
    if (nay(newline_index == 5)) k = CRC32_8(k, data[4]);
    return k % NUM_LOOKUP_TABLE_SIZE;
}

// Index by str len, hence we'll index from 1.
uint64_t hash_masks[101];
uint64_t hash_masks2[101];

// TODO: flexible simd
bool compare_strings(const char* source, size_t source_len, const char* target) {
  if (nay(source_len > 16)) {
    return memcmp(source, target, source_len) == 0;
  }
  uint64_t s1 = ((uint64_t*)source)[0] & hash_masks[source_len];
  uint64_t s2 = ((uint64_t*)source)[1] & hash_masks2[source_len];
  uint64_t t1 = ((uint64_t*)target)[0];
  uint64_t t2 = ((uint64_t*)target)[1];
  return s1 == t1 && s2 == t2;
}

// L1d: 48k data per physical core.
// Each cache line is 64 bytes wide.
// Each entry will take a total of 2 cache lines (117/128 bytes).
struct alignas(64) MinMaxAvg {
    int16_t min;
    int16_t max;
    unsigned int count;
    int64_t sum;
    alignas(8) char name[101] = {};
    char padding[11];

    MinMaxAvg() : min(std::numeric_limits<int16_t>::max()), max(std::numeric_limits<int16_t>::min()), sum(0), count(0) {}

    inline void update(std::string_view const& key_str, int _key, int16_t value) {
      // TODO: simd copy
        if (nay(name[0] == 0)) memcpy(name, key_str.data(), key_str.length());
        // debug(name);
        min = std::min(min, value);
        max = std::max(max, value);
        sum += value;
        ++count;
    }

    double Min() const {
        return (double)min / 10;
    }

    double Max() const {
        return (double)max / 10;
    }

    double avg() const {
        return count == 0 ? 0 : round((double)sum / count) / 10;
    }
};

static_assert(sizeof(MinMaxAvg) == 128, "MinMaxAvg not 128 bytes.");

// 16384 entries times 128 bytes is 2MB.
// L2 cache is 2mb per core so it should fit.
constexpr size_t HashMapSize = 1024 * 16;
using MapIndex = uint64_t;
using TheMap = std::array<MinMaxAvg, HashMapSize>;
static_assert(sizeof(TheMap) == 1024 * 16 * 128, "TheMap bad size.");
inline MinMaxAvg& lookup(TheMap& map, MapIndex key, std::string_view const& key_str) {
    auto lookup_key = key % HashMapSize;
    auto* entry = &map[lookup_key];
    // While we have bucket collison, linear probe forward while there is name that doesn't match.
    while (nay(entry->name[0] != 0 && !compare_strings(key_str.data(), key_str.length(), entry->name))) {
    // while (nay(entry->key && entry->key != key)) {
      // std::cout << "lookup collision: " << key_str << " with " << entry->name << std::endl;
      lookup_key = (lookup_key + 1) % HashMapSize;
      entry = &map[lookup_key];
    }
    return *entry;
}

inline uint32_t hash_name(std::string_view const& name) {
    uint32_t key=0;
    auto len = name.size();
    auto data = ((uint64_t*)name.data());
    auto key1 = data[0] & hash_masks[len];
    auto key2 = data[1] & hash_masks2[len];
    key = CRC32_64(0, key1);
    key = CRC32_64(key, key2);
    if (nay(len > 16)) {
      // Names longer than 16 bytes are an edge case.
      for (int i=16; i<len; ++i) {
        key = CRC32_8(key, name[i]);
      }
    }
    return key;
}

void init_tables() {
    // Compute number lookup table.
    for (int num = 0; num <= 99; ++num) {
        for (int decimal = 0; decimal <= 9; ++decimal) {
            auto s = std::to_string(num) + "." + std::to_string(decimal) + '\n';
            auto k = gen_num_key(s.data(), s.find('\n'));
            if (nay(num_lookup[k] != 0)) {
              std::cout << "collision! " << k << " " << num_lookup[k] << std::endl;
              exit(1);
            }
            num_lookup[k] = num * 10 + decimal;
            // std::cout << s.substr(0, s.size() - 1) << " " << k << " " << num_lookup[k] << std::endl;

            // Neg
            s = "-" + s;
            k = gen_num_key(s.data(), s.find('\n'));
            if (nay(num_lookup[k] != 0)) {
              std::cout << "collision! " << k << " " << num_lookup[k] << std::endl;
              exit(1);
            }
            num_lookup[k] = -num * 10 - decimal;
            // std::cout << s.substr(0, s.size() - 1) << " " << k << " " << num_lookup[k] << std::endl;
        }
    }
    // exit(0);

    // Index by str len, hence we'll index from 1.
    for (int i=1; i<=100; ++i) {
      hash_masks[i] = ~0x0;
      hash_masks2[i] = ~0x0;
    }
    for (int i=1; i<8; ++i) {
      // Discard tail bytes of first word.
      hash_masks[i] = (1ULL << (i * 8)) - 1;
      // Discard all of second word.
      hash_masks2[i] = 0;
    }
    for (int i=8; i<16; ++i) {
      // Discard tail bytes of second word.
      hash_masks2[i] = (1ULL << ((i-8) * 8)) - 1;
    }
    // DEBUG HASH MASKS.
    // for (int i=1; i<101; ++i) {
    //   std::cout << std::setw(16) << std::setfill('0') << std::hex << hash_masks2[i] <<
    //     std::setw(16) << std::setfill('0') << std::hex << hash_masks[i] << std::endl;
    // }
}

int main(int argc, char** argv) {
    auto t0 = Timer();
    std::vector<std::string> args(argv, argv + argc);
    std::string filename = args.size() > 1 ?  args[1] : "measurements.txt";

    // Determine the number of CPUs
    char* threads_str = std::getenv("THREADS");
    unsigned num_cpus = threads_str != nullptr ? std::atoi(threads_str) : std::thread::hardware_concurrency() / 2;

    char* chunks_str = std::getenv("CHUNKS");
    unsigned MAP_CHUNKS = chunks_str != nullptr ? std::atoi(chunks_str) : 8;

    size_t page_size = sysconf(_SC_PAGE_SIZE);

    // Open the file
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Error getting file size" << std::endl;
        close(fd);
        return 1;
    }

    // Memory map the file.
    char *map = static_cast<char*>(mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (map == MAP_FAILED) {
        std::cerr << "Error mapping file" << std::endl;
        close(fd);
        return 1;
    }
    close(fd);

    // Calculate start/end positions for each thread.
    // madvise(map, sb.st_size, MADV_RANDOM);
    std::vector<size_t> file_positions(num_cpus*MAP_CHUNKS+1, 0);
    size_t num_chunks = num_cpus * MAP_CHUNKS;
    size_t chunk_size = sb.st_size / num_chunks;
    for (unsigned i = 1; i < num_chunks; ++i) {
        size_t pos = i * chunk_size;
        while (map[pos-1] != '\n') ++pos;
        file_positions[i] = pos;
    }
    file_positions[num_chunks] = sb.st_size;

    // DEBUG START/END POSITIONS.
    // for (auto f : file_positions) {
    //   std::cout << f << std::endl;
    // }

    init_tables();

    std::atomic<int> counter(0);

    // Main processing function for each thread.
    auto process_chunk = [&](char* it, char* end, TheMap &result, int i, int c) {
      Timer t;
        // size_t pos=0;
        // char* it = buf.begin();
        // char* end = buf.end();
        int inner_counter = 0;

        const VecType semicolon = VEC_SET1_EPI8(';');
        char* line_start = it;

        while (yay(it < end)) {
            // Load unaligned bytes into simd register.
            VecType data = VEC_LOADU_SI(reinterpret_cast<const VecType*>(it));
            // DEBUG PRINT DATA
            // alignas(VecSize) char buffer[VecSize];
            // VEC_STORE_SI(reinterpret_cast<VecType*>(buffer), data);
            // std::string str(buffer, VecSize);
            // std::replace(str.begin(), str.end(), '\n', ' ');
            // std::cout << str << std::endl;
            // if (inner_counter % 1000000 == 0) std::cout << inner_counter << std::endl;

            // Compute mask of semicolon locations.
            MaskType sc_mask = VEC_CMPEQ_EPI8_MASK(data, semicolon);
            // std::bitset<VecSize> bs(sc_mask);
            // auto bs_str = bs.to_string();
            // debug(std::string(bs_str.rbegin(),  bs_str.rend()));

            // Loop once for each found semicolon.
            while (yay(sc_mask)) {
              int sc_index = TZCNT(sc_mask);
              char* sc_pos = it + sc_index;
              // Don't process semicolons meant for other threads (over-read).
              if (nay(sc_pos >= end)) break;

              // This is our station name.
              std::string_view key_str(line_start, sc_pos - line_start);
              // Compute key for name.
              uint64_t key = hash_name(key_str);
              auto& entry = lookup(result, key, key_str);

              // Extract value.
              char* v_pos = sc_pos + 1;
              int nl_index = 4;
              nl_index += v_pos[5] == '\n';
              nl_index -= v_pos[3] == '\n';

              auto num_key = gen_num_key(v_pos, nl_index);
              auto value = num_lookup[num_key];

              // std::cout << "len: " << key_str.size()  << " n: " << key_str << " k: " << key << " v: " << value << std::endl;

              // Locate entry in hashmap and update.
              entry.update(key_str, key, value);

              // Remove this semicolon from the semicolon mask and loop back around.
              sc_mask &= ~(1ULL << sc_index);
              line_start = v_pos + nl_index + 1;
              ++inner_counter;
              // if (inner_counter > 1000) exit(0);
            }

            it += VecSize;

            // pos += VecSize;
            // const size_t UNMAP_SIZE = 1024*1024*100;
            // // Leave 1 page to back ref line_start.
            // const size_t TRAIL = page_size;
            // if (nay(pos > TRAIL && ((pos-TRAIL) % (UNMAP_SIZE)) == 0)) {
            //   buf.unmap(pos-TRAIL);
            // }
        }
        counter += inner_counter;

        debug(c, " thread ", i, " complete ", t.milliseconds());
    };

    info(t0.milliseconds());
    std::vector<TheMap> thread_results(num_cpus);
    info("alloc thread results ", t0.milliseconds());

    std::unordered_map<std::string_view, MinMaxAvg> combined;
    combined.reserve(HashMapSize);
    info("alloc combined ", t0.milliseconds());

    ThreadPool pool(num_cpus);
    std::vector<std::unique_ptr<TaskManager>> chunk_tasks;

    ThreadPool unmap_pool(1, num_cpus);
    TaskManager unmap_tasks(unmap_pool);

    // std::vector<std::thread> unmap_threads(MAP_CHUNKS);
      // std::cout << "Setup: " << t0.milliseconds() << std::endl;
    auto t1 = Timer();

    for (int c=0; c<MAP_CHUNKS; ++c) {
      Timer t;

      auto& tasks = *chunk_tasks.emplace_back(std::make_unique<TaskManager>(pool));
      for (unsigned i = 0; i < num_cpus; ++i) {
          auto start_offset = file_positions[c*num_cpus+i];
          auto end_offset = file_positions[c*num_cpus+i+1];
          // debug("chunk: ", c, " thread: ", i, " so: ", start_offset, " eo: ", end_offset);

          tasks.enqueue([=, &thread_results] (int worker_thread) {
            process_chunk(map + start_offset, map + end_offset, std::ref(thread_results[worker_thread]), i, c);
        });
      }

      // tasks.flush();

      // Unmap in background.
      unmap_tasks.enqueue([=, &file_positions, &tasks] (int) {
        auto start_unmap = file_positions[c*num_cpus];
        auto end_unmap = file_positions[(c*num_cpus)+num_cpus] - page_size;
        start_unmap -= (start_unmap % page_size);
        debug(c, " unmapper waiting for tasks to complete");
        tasks.flush();
        Timer t;
        debug(c, " unmapping: ", start_unmap, " ", end_unmap);
        auto r = munmap(map + start_unmap, end_unmap - start_unmap);
        debug(c, " unmap done ", r, " ", t.milliseconds());
      });
    }
    // Wait for processing of all chunks to complete.
    for (auto& tasks : chunk_tasks) {
      tasks->flush();
    }
    auto t1r = t1.milliseconds();

    // Combine results.
    // WARNING: Loops over the entire table, assumes it isn't too huge.
    auto t2 = Timer();
    for (int i=0; i<num_cpus; ++i) {
      auto& thread_result = thread_results[i];
        for (auto &kv : thread_result) {
          if (!kv.name[0]) continue;
          auto& cr = combined[kv.name];
          if (kv.min < cr.min) cr.min = kv.min;
          if (kv.max > cr.max) cr.max = kv.max;
          cr.sum += kv.sum;
          cr.count += kv.count;
        }
    }
    auto t2r = t2.milliseconds();

    // Write to file.
    auto t3 = Timer();
    std::vector<std::string_view> keys;
    for (const auto &kv : combined) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());

    std::ofstream outFile("output.txt");
    outFile << std::fixed << std::setprecision(1) << "{";
    for (auto it = keys.begin(); it != keys.end(); ++it) {
      const auto &mav = combined[*it];
      outFile << *it << "=" << mav.Min() << "/" << mav.avg() << "/" << mav.Max();
      if (nay(std::next(it) != keys.end())) outFile << ", ";
    }
    outFile << "}\n";
    outFile.close();
    auto t3r = t3.milliseconds();

    debug("waiting unmap completion...");
    Timer t3_1;
    unmap_tasks.flush();
    auto t3_1r = t3_1.milliseconds();
    debug("unmap completion done... ", t3_1r);

    // Cleanup
    auto t4 = Timer();
    keys.clear();
    combined.clear();
    thread_results.clear();
    auto t4r = t4.milliseconds();
    auto total = t0.milliseconds();

    info("   Chunks: ", MAP_CHUNKS);
    info("  Threads: ", num_cpus);
    info(" Parallel: ", t1r);
    info("Combining: ", t2r);
    info("  Writing: ", t3r);
    info("    Unmap: ", t3_1r);
    info("    Clean: ", t4r);
    info("    Total: ", total);
    info("Processed: ", counter);

    return 0;
}
