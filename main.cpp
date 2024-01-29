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

#include <cstdio>
#include <ctime>
#include <string>
#include <sys/resource.h>
#include <sys/time.h>

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
template <typename... Args> void info(Args... args)
{
    std::ios_base::fmtflags f(std::cout.flags());
    ((std::cout << args), ...) << std::endl;
    std::cout.flags(f);
}

class data_buffer {
public:
    data_buffer(const std::string& filename, size_t start_offset, size_t length, int i)
        : _i(i)
    {
        // Open file
        int fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1) {
            std::cerr << "Error opening file" << std::endl;
            return;
        }

        // Memory map the file
        size_t page_size = sysconf(_SC_PAGE_SIZE);
        size_t aligned_offset = start_offset - (start_offset % page_size);
        size_t offset_difference = start_offset - aligned_offset;
        size_t adjusted_length = length + offset_difference;
        adjusted_length += page_size - (adjusted_length % page_size);
        size_t address = 0x7f0000000000ULL + 1024ULL*1024*300*i;
        _mapped_start = static_cast<char*>(mmap((void*)address, adjusted_length, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, aligned_offset));
        if (_mapped_start == MAP_FAILED || _mapped_start != (char*)address) {
            info("Error mapping file: ", strerror(errno), " requested: ", std::hex, (uintptr_t)address, " actual: ", (uintptr_t)_mapped_start);
            exit(1);
        }
        _aligned_start = _mapped_start;
        _start = _aligned_start + offset_difference;
        _end = _start + length;
        _aligned_end = _aligned_start + adjusted_length;

        // Close the file as it is no longer needed
        close(fd);

        // Map in a zero page.
        int zero_fd = open("/dev/zero", O_RDWR);
        if (zero_fd == -1) {
            info("Error opening /dev/zero");
            exit(1);
        }
        char* padding = static_cast<char*>(mmap(_aligned_end, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, zero_fd, 0));
        if (padding == MAP_FAILED || padding != _aligned_end) {
            info("Error mapping /dev/zero: ", strerror(errno), " requested: ", std::hex, (uintptr_t)_aligned_end, " actual: ", (uintptr_t)padding);
            exit(1);
        }
        close(zero_fd);

        // info(i, " file buf range: 0x",
        //     std::hex,
        //     (uintptr_t)_aligned_start, "-0x",
        //     (uintptr_t)_start, "-0x",
        //     (uintptr_t)_end, "-0x",
        //     (uintptr_t)(_aligned_end),
        //     std::dec,
        //     " offset: ", std::setw(8), start_offset,
        //     " length: ", std::setw(8), length);
        // info("dd if=", filename, " bs=1 skip=", aligned_offset, " count=", length+offset_difference, " | hexdump -C | less");

        // Remove mem protection on one page past the end of the mapping for overreads.
        // if (mprotect(_aligned_end, page_size, PROT_READ | PROT_WRITE | PROT_EXEC) == -1) {
        //     info("Error changing page protection: ", strerror(errno), " ", std::hex, (uintptr_t)_aligned_end);
        //     // exit(1);
        // }
    }

    ~data_buffer() {
      munmap(_mapped_start, _aligned_end - _mapped_start);
    }

    char* get_pointer_at(size_t index, size_t len) {
        return _start + index;
    }

    void unmap(size_t pos) {
        Timer t;
        auto r = munmap(_mapped_start, pos);

        // auto len = pos - (_mapped_start - _aligned_start);
        // auto r = munmap(_mapped_start, len);
        // info(std::dec, _i, " pos: ", pos, " unmapped: ", r, " ", std::hex, 
        //     (uintptr_t)_mapped_start, "-",
        //     (uintptr_t)_mapped_start + pos, std::dec, " ", t.milliseconds());
        // _mapped_start += len;
    }

    char* begin() {
        return _start;
    }

    char* end() {
        return _end;
    }

private:
    int _i;
    char* _mapped_start;
    char* _aligned_start;
    char* _start;
    char* _end;
    char* _aligned_end;
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
        // info(name);
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
    unsigned num_cpus = 64;//threads_str != nullptr ? std::atoi(threads_str) : std::thread::hardware_concurrency() / 2;

    size_t page_size = sysconf(_SC_PAGE_SIZE);

    // Calculate start/end positions for each thread.
    std::vector<size_t> file_positions(num_cpus+1, 0);
    {
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
      madvise(map, sb.st_size, MADV_RANDOM);

      size_t chunk_size = sb.st_size / num_cpus;
      for (unsigned i = 1; i < num_cpus; ++i) {
          size_t pos = i * chunk_size;
          while (map[pos-1] != '\n') ++pos;
          file_positions[i] = pos;
      }
      file_positions[num_cpus] = sb.st_size;

      munmap(map, sb.st_size);
    }
    // DEBUG START/END POSITIONS.
    // for (auto f : file_positions) {
    //   std::cout << f << std::endl;
    // }

    init_tables();

    std::atomic<int> counter(0);

    // DEBUG SIZE OF HASHMAP
    // std::cout << sizeof(MinMaxAvg) << std::endl;
    // exit(0);

    // Main processing function for each thread.
    auto process_chunk = [&](data_buffer& buf, TheMap &result, int i) {
        // size_t pos=0;
        char* it = buf.begin();
        char* end = buf.end();
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
            // info(std::string(bs_str.rbegin(),  bs_str.rend()));

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

        // info("Thread ", i, " complete");
    };

    // Launch threads
    std::vector<std::unique_ptr<data_buffer>> bufs(num_cpus);
    std::vector<std::thread> threads;
    info(t0.milliseconds());
    // TheMap* thread_results;
    // (TheMap*)posix_memalign((void**)&thread_results, 128, sizeof(TheMap) * num_cpus);
    // TheMap* thread_results = (TheMap*)aligned_alloc(128, sizeof(TheMap) * num_cpus);
    // TheMap* thread_results = (TheMap*)malloc(sizeof(TheMap) * num_cpus);
    // TheMap* thread_results = new TheMap[num_cpus];
    info(t0.milliseconds());
    // memset(thread_results, 0, sizeof(TheMap) * num_cpus);
    std::vector<TheMap> thread_results(num_cpus);
    info(t0.milliseconds());
    // info(std::hex, (uintptr_t)thread_results.data());
    auto t1 = Timer();
    for (unsigned i = 0; i < num_cpus; ++i) {
        auto start_offset = file_positions[i];
        auto lfb = std::make_unique<data_buffer>(filename, start_offset, file_positions[i+1] - start_offset, i);
        bufs[i] = std::move(lfb);
    }

    std::cout << "Setup: " << t0.milliseconds() << std::endl;

    for (unsigned i = 0; i < num_cpus; ++i) {
        // info("Pinning to cpus, logic: ", logic_cpu, " data: ", data_cpu);
        threads.emplace_back([=, &thread_results, &bufs] {
#if defined(__x86_64__)
          // Set thread affinity to CPU core 'i'.
          cpu_set_t cpuset;
          CPU_ZERO(&cpuset);
          CPU_SET(i, &cpuset);

          int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
          if (rc != 0) {
              std::cerr << "Error setting thread affinity: " << rc << std::endl;
          }
#endif
          // Call the processing function.
          process_chunk(*bufs[i], std::ref(thread_results[i]), i);
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

    // Cleanup
    auto t4 = Timer();
    keys.clear();
    combined.clear();
    // delete[] thread_results;
    // free(thread_results);
    thread_results.clear();
    bufs.clear();
    // munmap(map, sb.st_size);
    // close(fd);
    auto t4r = t4.milliseconds();
    auto total = t0.milliseconds();

    std::cout << "  Threads: " << num_cpus << std::endl;
    std::cout << " Parallel: " << t1r << std::endl;
    std::cout << "Combining: " << t2r << std::endl;
    std::cout << "  Writing: " << t3r << std::endl;
    std::cout << "    Clean: " << t4r << std::endl;
    std::cout << "    Total: " << total << std::endl;
    std::cout << "Processed: " << counter << std::endl;

    return 0;
}
