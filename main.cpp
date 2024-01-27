#include <cstdint>
#include <cmath>
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


// class lock_free_buffer {
// public:
//     lock_free_buffer(const std::string& filename, size_t start_offset, size_t length, int i)
//         : _spin_count(0)
//         , _file(filename, std::ios::binary)
//         , _bytes_read(0)
//         , _total_to_read(length)
//     {
//         // Alloc a buffer. round up to a neat simd vector size.
//         auto rounded_size = (length + VecSize - 1) & ~(VecSize - 1);
//         _start = (char*)std::aligned_alloc(VecSize, rounded_size);
//         // 0 set the padding.
//         memset(_start + length, 0, rounded_size - length);
//         // info("file buf range: 0x",
//         //     std::hex,
//         //     (uintptr_t)_start, "-0x",
//         //     (uintptr_t)_start + length, "-0x",
//         //     (uintptr_t)_start + rounded_size,
//         //     std::dec,
//         //     " offset: ", std::setw(8), start_offset,
//         //     " length: ", std::setw(8), length,
//         //     " padding: ", std::setw(2), rounded_size - length,
//         //     " alloc_size: ", rounded_size,
//         //     " simd_blocks: ", rounded_size / VecSize);
//         // info("dd if=", filename, " bs=1 skip=", start_offset, " count=", length, " | hexdump -C | less");
//
//         _file.seekg(start_offset, std::ios::beg);
//         _end = _start + length;
//         _write_position = _start;
//
//         read_chunk(1024*4);
//
//         // info("start: ", start_offset, " length: ", length);
//         // _thread = std::thread(&lock_free_buffer::read_file, this);
//         _thread = std::thread([=] {
//           // Set thread affinity to CPU core 'i'.
//           cpu_set_t cpuset;
//           CPU_ZERO(&cpuset);
//           CPU_SET(i, &cpuset);
//
//           int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
//           if (rc != 0) {
//               std::cerr << "Error setting thread affinity: " << rc << std::endl;
//           }
//
//           read_file();
//       });
//     }
//
//     ~lock_free_buffer() {
//         _thread.join();
//         free();
//     }
//
//     void free() {
//       std::free(_start);
//     }
//
//     char* get_pointer_at(size_t index, size_t len) {
//         char* start = _start + index;
//         char* end = std::min(start + len, _end);
//         // std::cout << "enter spin" << std::endl;
//         while (_write_position.load(std::memory_order_acquire) < end) {
//             // Spin wait
//             _spin_count++;
//         }
//         // std::cout << "exit spin" << std::endl;
//         return start;
//     }
//
//     char* end() {
//       return _end;
//     }
//
//     uint64_t get_spins() const {
//       return _spin_count;
//     }
//
// private:
//     void read_chunk(size_t chunk) {
//         size_t bytes_to_read = std::min(chunk, _total_to_read - _bytes_read);
//         _file.read(_start + _bytes_read, bytes_to_read);
//         _bytes_read += bytes_to_read;
//         _write_position.store(_start + _bytes_read, std::memory_order_release);
//     }
//
//     void read_file() {
//       auto t = Timer();
//         // read_chunk(1024*1024*10, bytes_read, total_to_read);
//         // info("starting read from: ", _file.tellg(), " to location ", std::hex, (uintptr_t)_start);
//
//         while (_bytes_read < _total_to_read) {
//             read_chunk(1024*4);
//             // if (bytes_read % (1024*1024*1024) == 0) {
//             //   std::cout << "read: " << bytes_read << std::endl;
//             // }
//         }
//         // info("read done: ", t.milliseconds(), "ms");
//     }
//
//     uint64_t _spin_count;
//     std::thread _thread;
//     std::ifstream _file;
//     char* _start;
//     char* _end;
//     std::atomic<char*> _write_position;
//     size_t _bytes_read;
//     size_t _total_to_read;
// };

class lock_free_buffer {
public:
    lock_free_buffer(const std::string& filename, size_t start_offset, size_t length, int i)
        : _start(nullptr)
        , _map(nullptr)
        , _end(nullptr)
        , _total_to_read(length)
        , _unmap_pos(0)
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
        // adjusted_length = (adjusted_length + page_size - 1) / page_size * page_size;
        adjusted_length = ((adjusted_length + page_size - 1) / page_size + 1) * page_size;
        _map = static_cast<char*>(mmap(nullptr, adjusted_length, PROT_READ, MAP_PRIVATE, fd, aligned_offset));
        if (_start == MAP_FAILED) {
            std::cerr << "Error mapping file: " << strerror(errno) << std::endl;
            _start = nullptr;
            close(fd);
            return;
        }
        _start = _map + offset_difference;
        _end = _start + length;

        // Close the file as it is no longer needed
        close(fd);

        info(i, " file buf range: 0x",
            std::hex,
            (uintptr_t)_map, "-0x",
            (uintptr_t)_start, "-0x",
            (uintptr_t)_end,
            // (uintptr_t)_start + rounded_size,
            std::dec,
            " offset: ", std::setw(8), start_offset,
            " length: ", std::setw(8), length);
            // " padding: ", std::setw(2), rounded_size - length,
            // " alloc_size: ", rounded_size,
            // " simd_blocks: ", rounded_size / VecSize);
        info("dd if=", filename, " bs=1 skip=", aligned_offset, " count=", length+offset_difference, " | hexdump -C | less");
    }

    ~lock_free_buffer() {
      munmap(_map, _total_to_read);
    }

    char* get_pointer_at(size_t index, size_t len) {
        return _start + index;
    }

    void unmap(size_t pos) {
        // Timer t;
        auto r = munmap(_map, pos);
        // info("unmapped: ", r, " ", pos, " ", t.milliseconds());
    }

    char* end() {
        return _end;
    }

private:
    size_t _unmap_pos;
    std::atomic_size_t _processed_length;
    char* _start;
    char* _map;
    char* _end;
    size_t _total_to_read;
    std::thread _thread;
    std::atomic<bool> _running=true;
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

  // while (source[i] != 0 && source[i] == target[i]) ++i;
  // return i == source_len;

    // return memcmp(source, target, source_len) == 0;
    // constexpr int block_size = 32;  // Size of SIMD register block
    //
    // for (int i = 0; i < source_len; i += block_size) {
    //     // Load block from source
    //     __m256i block_source = _mm256_loadu_si256((__m256i*)(source + i));
    //
    //     // Apply mask to the last block if source length is not a multiple of 32
    //     if (i + block_size > source_len) {
    //         __m256i mask;
    //         char temp_mask[block_size];
    //         for (int j = 0; j < block_size; ++j) {
    //             temp_mask[j] = (i + j < source_len) ? 0xFF : 0x00;
    //         }
    //         mask = _mm256_loadu_si256((__m256i*)temp_mask);
    //         block_source = _mm256_and_si256(block_source, mask);
    //     }
    //
    //     // Load block from target
    //     __m256i block_target = _mm256_loadu_si256((__m256i*)(target + i));
    //
    //     // Compare blocks
    //     __m256i result = _mm256_cmpeq_epi8(block_source, block_target);
    //
    //     // Check if all bytes in the result are 1s (indicating a match)
    //     if (!_mm256_testc_si256(result, _mm256_set1_epi8(-1))) {
    //         return false;  // Mismatch found
    //     }
    // }
    //
    // return true;  // All blocks matched
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

// bool compare_512bit_strings(const char* str1, const char* str2) {
//     __m512i v1 = _mm512_loadu_si512((const void*)str1);
//     __m512i v2 = _mm512_loadu_si512((const void*)str2);
//     __mmask64 result = _mm512_cmpeq_epi8_mask(v1, v2);
//     return result == 0xFFFFFFFFFFFFFFFF;
// }

// 16384 entries times 128 bytes is 2MB.
// L2 cache is 2mb per core so it should fit.
constexpr size_t HashMapSize = 1024 * 16;
using MapIndex = uint64_t;
using TheMap = std::array<MinMaxAvg, HashMapSize>;
inline MinMaxAvg* lookup(TheMap& map, MapIndex key, std::string_view const& key_str) {
    auto lookup_key = key % HashMapSize;
    auto* entry = &map[lookup_key];
    // While we have bucket collison, linear probe forward while there is name that doesn't match.
    while (nay(entry->name[0] != 0 && !compare_strings(key_str.data(), key_str.length(), entry->name))) {
    // while (nay(entry->key && entry->key != key)) {
      // std::cout << "lookup collision: " << key_str << " with " << entry->name << std::endl;
      lookup_key = (lookup_key + 1) % HashMapSize;
      entry = &map[lookup_key];
    }
    return entry;
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

int main(int argc, char** argv) {
    auto t0 = Timer();
    std::vector<std::string> args(argv, argv + argc);
    std::string filename = args.size() > 1 ?  args[1] : "measurements.txt";

    // Determine the number of CPUs
    char* threads_str = std::getenv("THREADS");
    unsigned num_cpus = threads_str != nullptr ? std::atoi(threads_str) : std::thread::hardware_concurrency() / 2;

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
    TheMap hash_map;
    using MapIndex = uint64_t;
    using TheMap = decltype(hash_map);
    // DEBUG SIZE OF HASHMAP
    // std::cout << sizeof(MinMaxAvg) << std::endl;
    // exit(0);

    // Main processing function for each thread.
    auto process_chunk = [&](lock_free_buffer& buf, TheMap &result, int i) {
        size_t pos=0;
        // +6 because we read the number past the semicolon.
        char* it = buf.get_pointer_at(pos, VecSize + 6);
        char* end = buf.end();
        int inner_counter = 0;

        const VecType semicolon = VEC_SET1_EPI8(';');
        char* line_start = it;

        while (yay(it < end)) {
            // Prefetch data to be accessed in the future (DOESN'T HELP).
            // _mm_prefetch(it + VecSize, _MM_HINT_T0);

            // Load aligned bytes into simd register.
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

              // Extract value.
              char* v_pos = sc_pos + 1;
              int nl_index = 4;
              nl_index += v_pos[5] == '\n';
              nl_index -= v_pos[3] == '\n';

              auto num_key = gen_num_key(v_pos, nl_index);
              auto value = num_lookup[num_key];

              // std::cout << "len: " << key_str.size()  << " n: " << key_str << " k: " << key << " v: " << value << std::endl;

              // Locate entry in hashmap and update.
              lookup(result, key, key_str)->update(key_str, key, value);

              // Remove this semicolon from the semicolon mask and loop back around.
              sc_mask &= ~(1ULL << sc_index);
              line_start = v_pos + nl_index + 1;
              ++inner_counter;
              // if (inner_counter > 1000) exit(0);
            }

            pos += VecSize;
            it += VecSize;
            // it = buf.get_pointer_at(pos, VecSize + 6);

            constexpr size_t UNMAP_SIZE = 1024*1024*10;
            if (nay((pos % (UNMAP_SIZE)) == 0)) {
              // Leave 1 page to back ref line_start.
              buf.unmap(pos-4096);
            }
        }
        counter += inner_counter;

        // info("Thread ", i, " spins: ", buf.get_spins());
    };

    std::cout << "Setup: " << t0.milliseconds() << std::endl;

    // Launch threads
    std::vector<std::unique_ptr<lock_free_buffer>> bufs(num_cpus);
    std::vector<std::thread> threads;
    std::vector<TheMap> thread_results(num_cpus);
    auto t1 = Timer();
    for (unsigned i = 0; i < num_cpus; ++i) {
        auto logic_cpu = i;
        auto start_offset = file_positions[i];
        auto lfb = std::make_unique<lock_free_buffer>(filename, start_offset, file_positions[i+1] - start_offset, logic_cpu);
        bufs[i] = std::move(lfb); // To free later.

        // info("Pinning to cpus, logic: ", logic_cpu, " data: ", data_cpu);
        threads.emplace_back([=, &thread_results, &bufs] {
#if defined(__x86_64__)
          // Set thread affinity to CPU core 'i'.
          cpu_set_t cpuset;
          CPU_ZERO(&cpuset);
          CPU_SET(logic_cpu, &cpuset);

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
    for (auto &thread_result : thread_results) {
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
    bufs.clear();
    thread_results.clear();
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
