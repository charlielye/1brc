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
#include <bitset>

#include <cstdio>
#include <ctime>
#include <string>
#include <sys/resource.h>
#include <sys/time.h>
#include <nmmintrin.h>
#include <immintrin.h>

/**
 * @brief Get the execution between a block of code.
 *
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

// A big 'ol lookup table.
int num_lookup[1<<20];

inline int gen_num_key_simple(const char* data, int newline_index) {
    // Potential value formats where n is newline.
    // -99.9n
    // -9.9n
    // 99.9n
    // 9.9n
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

inline int gen_num_key_simd(__m128i& data, int newline_index) {
    // Subtract 45 from the loaded data
    __m128i sub_45 = _mm_set1_epi8(45);
    auto sub_data = _mm_sub_epi8(data, sub_45);

    // Extract the individual characters after subtraction
    int it_0 = _mm_extract_epi8(sub_data, 0);
    int it_1 = _mm_extract_epi8(sub_data, 1);
    int it_2 = _mm_extract_epi8(sub_data, 2);
    int it_3 = _mm_extract_epi8(sub_data, 3);
    int it_4 = _mm_extract_epi8(sub_data, 4);

    // Check for newline characters
    int nl_g3 = newline_index > 3;
    int nl_5 = newline_index == 5;

    int k = it_0 << 16 | it_1 << 12 | it_2 << 8 | (it_3 << 4) * (nl_g3) | (it_4 * nl_5);
    return k;
}

struct MinMaxAvg {
    std::string_view name;
    int min;
    int max;
    int sum;
    unsigned int count;

    MinMaxAvg() : min(std::numeric_limits<int>::max()), max(std::numeric_limits<int>::min()), sum(0), count(0) {}

    void update(std::string_view const& key_str, int value) {
        name = key_str;
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

    // Loop for integers from -99 to 99
    for (int num = -99; num <= 99; ++num) {
        // Loop for decimal places from 0 to 9
        for (int decimal = 0; decimal <= 9; ++decimal) {
            auto s = std::to_string(num) + "." + std::to_string(decimal) + '\n';
            auto it = s.data();
            auto k = gen_num_key_simple(it, s.find('\n'));
            num_lookup[k] = num * 10 + (num < 0 ? -decimal : decimal);
            // std::cout  << k << " " << num_lookup[k] << std::endl;
        }
    }

    // Determine the number of CPUs
    unsigned num_cpus = 8;//std::thread::hardware_concurrency();

    // Create an array for starting positions
    std::vector<char*> starting_positions(num_cpus, map);

    // long pagesize = sysconf(_SC_PAGESIZE);

    // Calculate starting positions for each thread
    size_t chunk_size = sb.st_size / num_cpus;
    for (unsigned i = 1; i < num_cpus; ++i) {
        size_t pos = i * chunk_size;
        while (map[pos++] != '\n');
        auto start = map+pos;
        starting_positions[i] = start;

        // Advise the kernel of access pattern
        // char* aligned_start = reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(start) & ~(pagesize - 1));
        // if (madvise(aligned_start, chunk_size, MADV_SEQUENTIAL) != 0) {
        //     std::cerr << "Failed madvise." << std::endl;
        //     exit(-1);
        // }
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

    constexpr size_t HashMapSize = 1024 * 128;
    MinMaxAvg hash_map[HashMapSize];

    using MapIndex = uint64_t;
    using TheMap = decltype(hash_map);// std::unordered_map<MapIndex, MinMaxAvg>;

    // Function for each thread
    auto process_chunk = [&](char* start, char* end, TheMap &result) {
        char* it = start;
        int inner_counter = 0;

        const __m512i semicolon = _mm512_set1_epi8(';');
        const __m512i newline = _mm512_set1_epi8('\n');
        char* line_start = it;

        while (it < end) {
            // Load 64 bytes unaligned into 512 bit register.
            __m512i data = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(it));

            // DEBUG PRINT DATA
            // alignas(64) char buffer[64];
            // _mm512_store_si512(reinterpret_cast<__m512i*>(buffer), data); // Store the data from the SIMD register into the buffer
            // std::string str(buffer, 64); // Create a string from the buffer
            // std::replace(str.begin(), str.end(), '\n', ' ');
            // std::cout << str << std::endl;

            __mmask64 sc_mask = _mm512_cmpeq_epi8_mask(data, semicolon);
            

            while (sc_mask) {
              // std::bitset<64> bits(sc_mask);
              // std::cout << bits << std::endl;
              int sc_index = _tzcnt_u64(sc_mask);
              char* sc_pos = it + sc_index;
              if (__builtin_expect(sc_pos >= end, 0)) break;

              // Compute hash key for name.
              std::string_view key_str(line_start, sc_pos - line_start);
              // std::cout << key_str << std::endl;

              // WHY SO SLOW?
              // uint64_t key = 0;
              auto key1 = ((uint64_t*)line_start)[0];
              key1 &= hash_masks[key_str.size()];
              auto key2 = ((uint64_t*)line_start)[1];
              key2 &= hash_masks2[key_str.size()];
              // auto key = key1 ^ key2;
              uint64_t key = _mm_crc32_u64(0, key1);
              key = _mm_crc32_u64(key, key2);

              // std::cout << std::hex << key1 << " " << key2 << " " << key << " " << hash_masks[key_str.size()] << std::endl;
              // exit(0);

              // WORKS BIT SLOW: 1900
              // uint32_t key = 0;
              // switch (key_str.size()) {
              //   case 3:
              //     key = _mm_crc32_u16(key, *(uint16_t*)line_start);
              //     key = _mm_crc32_u8(key, line_start[2]);
              //     break;
              //   case 4:
              //     key = _mm_crc32_u32(key, *(uint32_t*)line_start);
              //     break;
              //   case 5:
              //     key = _mm_crc32_u32(key, *(uint32_t*)line_start);
              //     key = _mm_crc32_u8(key, line_start[4]);
              //     break;
              //   case 6:
              //     key = _mm_crc32_u32(key, *(uint32_t*)line_start);
              //     key = _mm_crc32_u16(key, *(uint16_t*)(line_start+4));
              //     break;
              //   case 7:
              //     key = _mm_crc32_u32(key, *(uint32_t*)line_start);
              //     key = _mm_crc32_u16(key, *(uint16_t*)(line_start+4));
              //     key = _mm_crc32_u8(key, line_start[6]);
              //     break;
              //   case 8:
              //     key = _mm_crc32_u64(key, *(uint64_t*)line_start);
              //     break;
              //   default:
              //     key = _mm_crc32_u64(key, *(uint64_t*)line_start);
              //     key = _mm_crc32_u8(key, line_start[8]);
              //     break;
              // }

              // ABOUT THE SAME: 1800
              // uint64_t key = 0;
              // for (int i=0; i<9 && i < key_str.size(); ++i) {
              //   key = (key * 31) ^ line_start[i];
              // }

              // ABOUT THE SAME: 1800
              // uint64_t key = 0;
              // for (int i=0; i<9 && i < key_str.size(); ++i) {
              //   key = (key * 31) ^ line_start[i];
              // }

              // ABOUT THE SAME
              // uint64_t key = 0;
              // for (int i=0; i<9 && i < key_str.size(); ++i) {
              //     key = _mm_crc32_u8(key, line_start[i]);
              // }

              // 2700
              // uint64_t data_aligned[2] = { 0 };
              // std::memcpy(data_aligned, line_start, std::min(key_str.size(), (size_t)9));
              // uint64_t key = _mm_crc32_u64(0, data_aligned[0]);
              // key = _mm_crc32_u64(key, data_aligned[1]);

              // Extract value.
              char* v_pos = sc_pos + 1;
              int nl_index = 4;
              if (v_pos[3] == '\n') nl_index = 3;
              if (v_pos[5] == '\n') nl_index = 5;
              auto num_key = gen_num_key_simple(v_pos, nl_index);
              auto value = num_lookup[num_key];

              result[key % HashMapSize].update(key_str, value);

              sc_mask &= ~(1ULL << sc_index);
              line_start = v_pos + nl_index + 1;
              ++inner_counter;
              // std::cout << "len: " << key_str.size()  << " n: " << key_str << " k: " << key << " v: " << value << std::endl;
              // if (inner_counter > 100) exit(0);
            }

            it += 64;
            // std::cout << inner_counter << std::endl;
            // exit(0);

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
        // thread_results[i].reserve(1024 << 3);
        threads.push_back(std::thread(process_chunk, start, end, std::ref(thread_results[i])));
    }

    // Join threads
    for (auto &t : threads) {
        t.join();
    }
    auto t1r = t1.milliseconds();

    // Combine results.
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

    std::cout << " Parallel: " << t1r << std::endl;
    std::cout << "Combining: " << t2r << std::endl;
    std::cout << "  Writing: " << t3r << std::endl;
    std::cout << "    Unmap: " << t4r << std::endl;
    std::cout << "    Total: " << total << std::endl;
    std::cout << "Processed: " << counter << std::endl;

    // for (int j=0; j<thread_results.size(); ++j) {
    //     size_t totalCollisions = 0;
    //     for (size_t i = 0; i < thread_results[j].bucket_count(); ++i) {
    //         size_t bucketSize = thread_results[j].bucket_size(i);
    //         if (bucketSize > 1) {
    //             totalCollisions += (bucketSize - 1);
    //             // std::cout << "Bucket " << i << " has " << bucketSize << " elements (collision!)\n";
    //         }
    //     }
    //
    //     std::cout << "T" << j << " Total collisions: " << totalCollisions << std::endl;
    //     std::cout << "T" << j << " Load factor: " << thread_results[j].load_factor() << std::endl;
    // }

    return 0;
}
