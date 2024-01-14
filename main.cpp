#include <algorithm>
#include <crc32intrin.h>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <string>
#include <fcntl.h>
#include <limits>

#include <cstdio>
#include <ctime>
#include <string>
#include <sys/resource.h>
#include <sys/time.h>
#include <nmmintrin.h>
#include <immintrin.h>

#ifdef __AVX512F__
using VecType = __m512i;
using MaskType = __mmask64;
const int VecSize = 64;
#define VEC_SET1_EPI8 _mm512_set1_epi8
#define VEC_LOADU_SI _mm512_loadu_si512
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
#define VEC_CMPEQ_EPI8_MASK(data, val) _mm256_movemask_epi8(_mm256_cmpeq_epi8(data, val))
#define VEC_STORE_SI(buffer, data) _mm256_store_si256(reinterpret_cast<__m256i*>(buffer), data)
#define TZCNT _tzcnt_u32
#endif


/**
 * Get the execution between a block of code.
 */
class Timer {
  private:
    struct timespec _startTime;
    struct timespec _endTime;

    static constexpr int64_t NanosecondsPerSecond = 1000LL * 1000 * 1000;

    /**
     * @brief Manually sets the start time.
     */
    void start() { clock_gettime(CLOCK_REALTIME, &_startTime); }

    /**
     * @brief Manually sets the end time.
     */
    void end() { clock_gettime(CLOCK_REALTIME, &_endTime); }

  public:
    /**
     * @brief Initialize a Timer with the current time.
     *
     */
    Timer()
        : _endTime({})
    {
        start();
    }

    /**
     * @brief Return the number of nanoseconds elapsed since the start of the timer.
     */
    [[nodiscard]] int64_t nanoseconds() const
    {
        struct timespec end;
        if (_endTime.tv_nsec == 0 && _endTime.tv_sec == 0) {
            clock_gettime(CLOCK_REALTIME, &end);
        } else {
            end = _endTime;
        }

        int64_t nanos = (end.tv_sec - _startTime.tv_sec) * NanosecondsPerSecond;
        nanos += (end.tv_nsec - _startTime.tv_nsec);

        return nanos;
    }

    /**
     * @brief Return the number of nanoseconds elapsed since the start of the timer.
     */
    [[nodiscard]] double milliseconds() const
    {
        int64_t nanos = nanoseconds();
        return static_cast<double>(nanos) / 1000000;
    }

    /**
     * @brief Return the number of seconds elapsed since the start of the timer.
     */
    [[nodiscard]] double seconds() const
    {
        int64_t nanos = nanoseconds();
        double secs = static_cast<double>(nanos) / NanosecondsPerSecond;
        return secs;
    }

    /**
     * @brief Return the number of seconds elapsed since the start of the timer as a string.
     */
    [[nodiscard]] std::string toString() const
    {
        double secs = seconds();
        return std::to_string(secs);
    }
};

// A big 'ol number lookup table.
int num_lookup[1<<20];

// Computes the key for looking up in the number lookup table.
// Potential value formats where n is newline.
// -99.9n
// -9.9n
// 99.9n
// 9.9n
inline int gen_num_key(const char* data, int newline_index) {
    int it_0 = (data[0] - 45) << 16;
    int it_1 = (data[1] - 45) << 12;
    int it_2 = (data[2] - 45) << 8;
    int it_3 = (data[3] - 45) << 4;
    int it_4 = data[4] - 45;

    int nl_g3 = newline_index > 3;
    int nl_5 = newline_index == 5;

    int k = it_0 | it_1 | it_2 | (it_3 * nl_g3) | (it_4 * nl_5);
    return k;
}

// inline int gen_num_key_simd(__m128i& data, int newline_index) {
//     // Subtract 45 from the loaded data
//     __m128i sub_45 = _mm_set1_epi8(45);
//     auto sub_data = _mm_sub_epi8(data, sub_45);
//
//     // Extract the individual characters after subtraction
//     int it_0 = _mm_extract_epi8(sub_data, 0);
//     int it_1 = _mm_extract_epi8(sub_data, 1);
//     int it_2 = _mm_extract_epi8(sub_data, 2);
//     int it_3 = _mm_extract_epi8(sub_data, 3);
//     int it_4 = _mm_extract_epi8(sub_data, 4);
//
//     // Check for newline characters
//     int nl_g3 = newline_index > 3;
//     int nl_5 = newline_index == 5;
//
//     int k = it_0 << 16 | it_1 << 12 | it_2 << 8 | (it_3 << 4) * (nl_g3) | (it_4 * nl_5);
//     return k;
// }

struct MinMaxAvg {
    std::string_view name;
    int min;
    int max;
    int sum;
    unsigned int count;

    MinMaxAvg() : min(std::numeric_limits<int>::max()), max(std::numeric_limits<int>::min()), sum(0), count(0) {}

    void update(std::string_view const& key_str, int value) {
        name = key_str;
        // BRANCHLESS IS WORSE.
        // bool lt = value < min;
        // bool gt = value > max;
        // min = min * !lt + value * lt;
        // max = max * !gt + value * gt;
        if (value < min) min = value;
        if (value > max) max = value;
        sum += value;
        ++count;
    }

    float Min() const {
        return (float)min / 10;
    }

    float Max() const {
        return (float)max / 10;
    }

    float avg() const {
        return count == 0 ? 0 : (float)sum / (count*10);
    }
};

int main(int argc, char** argv) {
    auto t0 = Timer();
    std::vector<std::string> args(argv, argv + argc);

    // Open the file
    std::string filename = args.size() > 1 ?  args[1] : "measurements.txt";
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

    // Memory map the file
    char *map = static_cast<char*>(mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (map == MAP_FAILED) {
        std::cerr << "Error mapping file" << std::endl;
        close(fd);
        return 1;
    }

    // Determine the number of CPUs
    unsigned num_cpus = 64;//std::thread::hardware_concurrency();

    // long pagesize = sysconf(_SC_PAGESIZE);

    // Calculate starting positions for each thread.
    std::vector<char*> starting_positions(num_cpus, map);
    size_t chunk_size = sb.st_size / num_cpus;
    for (unsigned i = 1; i < num_cpus; ++i) {
        size_t pos = i * chunk_size;
        while (map[pos++] != '\n');
        auto start = map+pos;
        starting_positions[i] = start;

        // Advise the kernel of access pattern (DOESN'T HELP).
        // char* aligned_start = reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(start) & ~(pagesize - 1));
        // if (madvise(aligned_start, chunk_size, MADV_SEQUENTIAL) != 0) {
        //     std::cerr << "Failed madvise." << std::endl;
        //     exit(-1);
        // }
    }

    // Compute number lookup table.
    for (int num = -99; num <= 99; ++num) {
        for (int decimal = 0; decimal <= 9; ++decimal) {
            auto s = std::to_string(num) + "." + std::to_string(decimal) + '\n';
            auto it = s.data();
            auto k = gen_num_key(it, s.find('\n'));
            num_lookup[k] = num * 10 + (num < 0 ? -decimal : decimal);
            // std::cout  << k << " " << num_lookup[k] << std::endl;
        }
    }

    // Index by str len, hence we'll index from 1.
    uint64_t hash_masks[101];
    uint64_t hash_masks2[101];
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

    std::atomic<int> counter(0);

    // ***********************************************************************
    // TODO: WARNING! WARNING! WARNING! THIS DOESN'T YET HANDLE COLLISIONS!
    // Magic number chosen to avoid collisions on test data set.
    // Use open addressing to handle collisions, avoid memory allocs.
    //
    //
    //
    //
    //
    //
    //
    //
    //
    // DID YOU FORGET ABOUT THIS!? DONT!
    // ***********************************************************************
    constexpr size_t HashMapSize = 1024 * 128;
    MinMaxAvg hash_map[HashMapSize];
    using MapIndex = uint64_t;
    using TheMap = decltype(hash_map);

    // Main processing function for each thread.
    auto process_chunk = [&](char* start, char* end, TheMap &result) {
        char* it = start;
        int inner_counter = 0;

        const VecType semicolon = VEC_SET1_EPI8(';');
        char* line_start = it;

        while (it < end) {
            // Prefetch data to be accessed in the future (DOESN'T HELP).
            // _mm_prefetch(it + VecSize, _MM_HINT_T0);

            // Load 64 bytes unaligned into 512 bit register.
            VecType data = VEC_LOADU_SI(reinterpret_cast<const VecType*>(it));

            // DEBUG PRINT DATA
            // alignas(VecSize) char buffer[VecSize];
            // VEC_STORE_SI(reinterpret_cast<__m512i*>(buffer), data);
            // std::string str(buffer, 64);
            // std::replace(str.begin(), str.end(), '\n', ' ');
            // std::cout << str << std::endl;

            // Compute mask of semicolon locations.
            MaskType sc_mask = VEC_CMPEQ_EPI8_MASK(data, semicolon);
            
            // Loop once for each found semicolon.
            while (sc_mask) {
              int sc_index = TZCNT(sc_mask);
              char* sc_pos = it + sc_index;
              // Don't process semicolons meant for other threads (over-read).
              if (__builtin_expect(sc_pos >= end, 0)) break;

              // This is our station name.
              std::string_view key_str(line_start, sc_pos - line_start);
              // std::cout << key_str << std::endl;

              // Compute key for name.
              // Names longer than 16 bytes are an edge case.
              uint64_t key = 0;
              if (__builtin_expect(key_str.size() <= 16, 1)) {
                auto key1 = ((uint64_t*)line_start)[0];
                key1 &= hash_masks[key_str.size()];
                auto key2 = ((uint64_t*)line_start)[1];
                key2 &= hash_masks2[key_str.size()];
                key = _mm_crc32_u64(0, key1);
                key = _mm_crc32_u64(key, key2);
              } else {
                for (int i=0; i < key_str.size(); ++i) {
                  key = _mm_crc32_u8(key, line_start[i]);
                }
              }

              // Extract value.
              char* v_pos = sc_pos + 1;
              int nl_index = 4;
              if (v_pos[3] == '\n') nl_index = 3;
              if (v_pos[5] == '\n') nl_index = 5;
              auto num_key = gen_num_key(v_pos, nl_index);
              auto value = num_lookup[num_key];

              // Locate entry in hashmap and update.
              // TODO: Handle collision.
              result[key % HashMapSize].update(key_str, value);

              // Remove this semicolon from the semicolon mask and loop back around.
              sc_mask &= ~(1ULL << sc_index);
              line_start = v_pos + nl_index + 1;
              ++inner_counter;
              // std::cout << "len: " << key_str.size()  << " n: " << key_str << " k: " << key << " v: " << value << std::endl;
              // if (inner_counter > 100) exit(0);
            }

            it += VecSize;
        }
        counter += inner_counter;
    };

    // Launch threads
    auto t1 = Timer();
    std::vector<std::thread> threads;
    std::vector<TheMap> thread_results(num_cpus);
    for (unsigned i = 0; i < num_cpus; ++i) {
        char* start = starting_positions[i];
        char* end = (i == num_cpus - 1) ? map + sb.st_size : starting_positions[i + 1];
        threads.emplace_back([=, &thread_results] {
          // Set thread affinity to CPU core 'i'
          cpu_set_t cpuset;
          CPU_ZERO(&cpuset);
          CPU_SET(i, &cpuset);

          int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
          if (rc != 0) {
              std::cerr << "Error setting thread affinity: " << rc << std::endl;
          }

          // Call the processing function
          process_chunk(start, end, std::ref(thread_results[i]));
      });
    }

    // Join threads
    for (auto &t : threads) {
        t.join();
    }
    auto t1r = t1.milliseconds();

    // Combine results.
    // WARNING: Loops over the entire table, assumes it isn't too huge.
    auto t2 = Timer();
    std::unordered_map<std::string_view, MinMaxAvg> combinedResults;
    combinedResults.reserve(1024);
    for (auto &thread_result : thread_results) {
        for (auto &kv : thread_result) {
          if (kv.name.empty()) continue;
          auto& cr = combinedResults[kv.name];
          if (kv.min < cr.min) cr.min = kv.min;
          if (kv.max > cr.max) cr.max = kv.max;
          cr.sum += kv.sum;
          cr.count += kv.count;

          // auto& cr = combinedResults[kv.second.name];
          // if (kv.second.min < cr.min) cr.min = kv.second.min;
          // if (kv.second.max > cr.max) cr.max = kv.second.max;
          // cr.sum += kv.second.sum;
          // cr.count += kv.second.count;
        }
    }
    auto t2r = t2.milliseconds();

    // Write to file.
    auto t3 = Timer();
    std::vector<std::string_view> keys;
    for (const auto &kv : combinedResults) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());

    std::ofstream outFile("output.txt");
    outFile << std::fixed << std::setprecision(1);
    for (const auto &key : keys) {
        const auto &mav = combinedResults[key];
        outFile << key << "=" << mav.Min() << "/" << mav.avg() << "/" << mav.Max() << std::endl;
    }
    outFile.close();
    auto t3r = t3.milliseconds();

    // Cleanup
    auto t4 = Timer();
    munmap(map, sb.st_size);
    close(fd);
    auto t4r = t4.milliseconds();
    auto total = t0.milliseconds();

    std::cout << "  Threads: " << num_cpus << std::endl;
    std::cout << " Parallel: " << t1r << std::endl;
    std::cout << "Combining: " << t2r << std::endl;
    std::cout << "  Writing: " << t3r << std::endl;
    std::cout << "    Unmap: " << t4r << std::endl;
    std::cout << "    Total: " << total << std::endl;
    std::cout << "Processed: " << counter << std::endl;

    return 0;
}
