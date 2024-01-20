#include <algorithm>
#include <cmath>
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

#define yay(x) __builtin_expect(x, 1)
#define nay(x) __builtin_expect(x, 0)

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

// A big 'ol number lookup table.
constexpr int NUM_LOOKUP_TABLE_SIZE = 1<<15;
int num_lookup[NUM_LOOKUP_TABLE_SIZE];

// Computes the key for looking up in the number lookup table.
// Potential value formats where n is newline.
// -99.9n
// -9.9n
// 99.9n
// 9.9n
inline __attribute__((always_inline)) int gen_num_key(const char* data, int newline_index) {
    uint32_t i = *(uint32_t*)(data);
    uint32_t k = _mm_crc32_u32(0, i);
    if (nay(newline_index == 5)) k = _mm_crc32_u8(k, data[4]);
    return k % NUM_LOOKUP_TABLE_SIZE;
}

struct MinMaxAvg {
    std::string_view name;
    uint32_t key;
    int16_t min;
    int16_t max;
    int64_t sum;
    unsigned int count;

    MinMaxAvg() : min(std::numeric_limits<int16_t>::max()), max(std::numeric_limits<int16_t>::min()), sum(0), count(0), key(0) {}

    inline void update(std::string_view const& key_str, int _key, int16_t value) {
        name = key_str;
        key = _key;
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

// 16384 entries times 32 bytes an entry is 524288 bytes.
constexpr size_t HashMapSize = 1024 * 16;
using MapIndex = uint64_t;
using TheMap = MinMaxAvg[HashMapSize];
inline MinMaxAvg* lookup(TheMap& map, MapIndex key) {
    auto lookup_key = key % HashMapSize;
    auto* entry = &map[lookup_key];
    while (nay(entry->key && entry->key != key)) {
      lookup_key = (lookup_key + 1) % HashMapSize;
      entry = &map[lookup_key];
    }
    return entry;
}

// Index by str len, hence we'll index from 1.
uint64_t hash_masks[101];
uint64_t hash_masks2[101];

inline  __attribute__((always_inline)) uint32_t hash_name(std::string_view const& name) {
    uint32_t key=0;
    auto len = name.size();
    auto data = ((uint64_t*)name.data());
    auto key1 = data[0] & hash_masks[len];
    auto key2 = data[1] & hash_masks2[len];
    key = _mm_crc32_u64(0, key1);
    key = _mm_crc32_u64(key, key2);
    if (nay(len > 16)) {
      // Names longer than 16 bytes are an edge case.
      for (int i=16; i<len; ++i) {
        key = _mm_crc32_u8(key, name[i]);
      }
    }
    return key;
}

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
    char* threads_str = std::getenv("THREADS");
    unsigned num_cpus = threads_str != nullptr ? std::atoi(threads_str) : std::thread::hardware_concurrency() / 2;

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

    std::atomic<int> counter(0);

    // 16384 entries times 32 bytes and entry is 524288 bytes.
    constexpr size_t HashMapSize = 1024 * 16;
    MinMaxAvg hash_map[HashMapSize];
    using MapIndex = uint64_t;
    using TheMap = decltype(hash_map);

    // std::cout << sizeof(MinMaxAvg) << std::endl;
    // exit(0);

    // Main processing function for each thread.
    auto process_chunk = [&](char* start, char* end, TheMap &result) {
        char* it = start;
        int inner_counter = 0;

        const VecType semicolon = VEC_SET1_EPI8(';');
        char* line_start = it;

        while (yay(it < end)) {
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
            while (yay(sc_mask)) {
              int sc_index = TZCNT(sc_mask);
              char* sc_pos = it + sc_index;
              // Don't process semicolons meant for other threads (over-read).
              if (nay(sc_pos >= end)) break;

              // This is our station name.
              std::string_view key_str(line_start, sc_pos - line_start);
              // std::cout << key_str << std::endl;
              // Compute key for name.
              uint64_t key = hash_name(key_str);

              // Extract value.
              char* v_pos = sc_pos + 1;
              int nl_index = 4;
              nl_index += v_pos[5] == '\n';
              nl_index -= v_pos[3] == '\n';
              auto num_key = gen_num_key(v_pos, nl_index);
              auto value = num_lookup[num_key];

              // Locate entry in hashmap and update.
              lookup(result, key)->update(key_str, key, value);

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
    std::unordered_map<std::string_view, MinMaxAvg> combined;
    combined.reserve(HashMapSize);
    for (auto &thread_result : thread_results) {
        for (auto &kv : thread_result) {
          if (!kv.key) continue;
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
