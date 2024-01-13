#include <algorithm>
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

// int ones_table[58] = {
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
//   2, 3, 4, 5, 6, 7, 8, 9 };
// int tens_table[58] = {
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 10,
//   20, 30, 40, 50, 60, 70, 80, 90 };
// int hundreds_table[58] = {
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//   0, 0, 0, 0, 0, 0, 0, 0, 0, 100,
//   200, 300, 400, 500, 600, 700, 800, 900 };

inline int gen_num_key(char*& it) {
    // Potential value formats where n is newline.
    // -99.9n
    // -9.9n
    // 99.9n
    // 9.9n

    // // Load 16 bytes from an aligned location
    // __m128i aligned_data = _mm_load_si128((__m128i*)(it - 1));  // Load an extra byte to ensure alignment
    //
    // // Extract the lower 16 - shift bytes (shift is the number of bytes to shift)
    // int shift = (uintptr_t)(it - 1) & 0xF;
    //
    // if (shift == 0) {
    //     // No shift required
    // } else if (shift == 1) {
    //     aligned_data = _mm_srli_si128(aligned_data, 1);
    // } else if (shift == 2) {
    //     aligned_data = _mm_srli_si128(aligned_data, 2);
    // } // Add more conditions as needed for other shift values
    //
    // // Subtract 45 from the loaded data
    // __m128i sub_45 = _mm_set1_epi8(45);
    // aligned_data = _mm_sub_epi8(aligned_data, sub_45);
    //
    // // Extract the individual characters after subtraction
    // int it_0 = _mm_extract_epi8(aligned_data, 0);
    // int it_1 = _mm_extract_epi8(aligned_data, 1);
    // int it_2 = _mm_extract_epi8(aligned_data, 2);
    // int it_3 = _mm_extract_epi8(aligned_data, 3);
    // int it_4 = _mm_extract_epi8(aligned_data, 4);
    //
    // // Check for newline characters
    // int nl_3 = _mm_extract_epi8(aligned_data, 3) == '\n';
    // int nl_4 = _mm_extract_epi8(aligned_data, 4) == '\n';
    // int nl_5 = _mm_extract_epi8(aligned_data, 5) == '\n';
    //
    // int value = 0;
    // int k = it_0 << 16 | it_1 << 12 | it_2 << 8 | (it_3 << 4) * (nl_4 | nl_5) | (it_4 * nl_5);
    // it += (5 * nl_4) + (4 * nl_3) + (6 * nl_5);
    // return k;

    int it_0 = *it;
    int it_1 = *(it + 1);
    int it_2 = *(it + 2);
    int it_3 = *(it + 3);
    int it_4 = *(it + 4);
    int nl_5 = *(it + 5) == '\n';
    int nl_4 = it_4 == '\n';
    int nl_3 = it_3 == '\n';
    it_0 -= 45;
    it_1 -= 45;
    it_2 -= 45;
    it_3 -= 45;
    it_4 -= 45;
    int value = 0;
    int k = it_0 << 16 | it_1 << 12 | it_2 << 8 | (it_3 << 4) * (nl_4 | nl_5) | (it_4 * nl_5);
    it += (5 * nl_4) + (4 * nl_3) + (6 * nl_5);
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
            auto k = gen_num_key(it);
            num_lookup[k] = num * 10 + (num < 0 ? -decimal : decimal);
            // std::cout  << k << " " << num_lookup[k] << std::endl;
        }
    }

    // Determine the number of CPUs
    unsigned num_cpus = 1;//std::thread::hardware_concurrency();

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

    std::atomic<int> counter(0);

    using MapIndex = uint64_t;
    using TheMap = std::unordered_map<MapIndex, MinMaxAvg>;

    // Function for each thread
    auto process_chunk = [map, &counter](char* start, char* end, TheMap &result) {
        char* it = start;
        int inner_counter = 0;

        alignas(16) std::aligned_storage_t<128, 16> _name_buffer;
        alignas(16) std::aligned_storage_t<16, 16> _value_buffer;
        auto name_buffer = (char*)&_name_buffer;
        auto value_buffer = (char*)&_value_buffer;
        const __m128i semicolon = _mm_set1_epi8(';');
        const __m128i newline = _mm_set1_epi8('\n');
        const __m128i zero = _mm_setzero_si128();

        // Each time around the loop, it points to the start of the next line.
        while (it < end) {
          char* aligned_start = (char*)((uintptr_t)it & ~(uintptr_t)0x0F);
          char* aligned_it = aligned_start;
          int offset = it - aligned_it;

          // Pos relative to it.
          bool first = true;
          // int semicolon_pos = -offset;
          // int newline_pos = 0;
          
          // Loop reading into data, until semicolon_index is set (relative to data).
          __m128i data;
          int semicolon_index = -1;
          while (true) {
outer:
            data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(aligned_it));

            __m128i result_semicolon = _mm_cmpeq_epi8(data, semicolon);

            int semicolon_mask = _mm_movemask_epi8(result_semicolon);

            if (!semicolon_mask) {
              // Gosh, it's a long name, or we're very mis-aligned. Keep on reading.
              aligned_it += 16;
              first = false;
              continue;
            }

            semicolon_index = __builtin_ctz(semicolon_mask);
            
            while (first && semicolon_index < offset) {
              // Case where our aligned read, read the previous line.
              // Our aligned read could return e.g.
              //
              // ch;27.2\nGabès;24.0
              //          ^-- Start of line.
              //
              // In this case, semicolon_index = 2, offset = 7.
              // Zero the bit and go again, after which semicolon_index = 13.
              semicolon_mask &= ~(1 << semicolon_index);
              // std::cout << "here " << offset << " " << semicolon_mask << std::endl;
              if (!semicolon_mask) {
                aligned_it += 16;
                first = false;
                goto outer;
              }
              semicolon_index = __builtin_ctz(semicolon_mask);
            }

            break;
          }

            // __m128i result_newline = _mm_cmpeq_epi8(data, newline);
            // int newline_mask = _mm_movemask_epi8(result_newline);
            // int newline_index = newline_mask ? __builtin_ctz(newline_mask) : -1;

            // We've found the semicolon delimeter, populate our name_buffer for hashing.
            // Erase up to 15 zeros after the name, so we can hash without collecting garbage.
            auto name_len = (aligned_it + semicolon_index) - aligned_start - offset;
            std::memcpy(name_buffer, it, name_len);
            std::memset(name_buffer + name_len, 0, 16);
            auto key = _mm_crc32_u64(0ULL, *reinterpret_cast<uint64_t*>(name_buffer));
            key = _mm_crc32_u64(key, *reinterpret_cast<uint64_t*>(name_buffer+8));

            std::string_view key_str(it, name_len);
            it += name_len + 1;
            // std::memcpy(value_buffer, it, 6);
            auto num_key = gen_num_key(it);
            auto value = num_lookup[num_key];
            // if (semicolon_index > 9) {
            //   // If our last buffer read has more than 9 name chars in it, then we may not have the full value.
            //   // _____name;-99.99\n
            // }

            // std::cout << "offset: " << offset << " len: " << name_len  << " n: " << name_buffer << " k: " << key << " v:" << value << std::endl;

            // while (*it != '\n') ++it;
            // ++it;

            // Update the result map
            result[key].update(key_str, value);

            ++inner_counter;
            // if (inner_counter > 1000) break;
            // continue;

            // std::memcpy(name_buffer, str + i, copyLength);


        // while (it < end) {
            // char* s=it;
            // // Gen key while finding the ';' character.
            // uint32_t key = 0;
            // // uint32_t key = 0xFFFFFFFF;
            // while (*it != ';') {
            //   // key = (key * 31) ^ (unsigned char)*it;
            //   key = _mm_crc32_u8(key, *it);
            //   ++it;
            // }
            // // key = ~key;
            // std::string_view key_str(s, it - s);
            //
            // ++it;

            // Potential value formats where n is newline and * is anything.
            // -99.9n
            // -9.9n
            // 99.9n
            // 9.9n*
            // int it_3 = *(it + 3);
            // int it_4 = *(it + 4);
            // int value = 0;
            // if (__builtin_expect(it_4 == '\n', 1)) {
            //   int it_0 = *it;
            //   int it_1 = *(it + 1);
            //   bool is_negative = (it_0 == '-');
            //   value = tens_table[it_1] + ones_table[it_3];
            //   value = -value * is_negative + (hundreds_table[it_0] + value) * !is_negative;
            //   it += 5;
            // } else {
            //   int it_0 = *it;
            //   int it_2 = *(it + 2);
            //   if (it_3 == '\n') {
            //     value = tens_table[it_0] + ones_table[it_2];
            //     it += 4;
            //   } else {
            //     int it_1 = *(it + 1);
            //     value = -(hundreds_table[it_1] + tens_table[it_2] + ones_table[it_4]);
            //     it += 6;
            //   }
            // }

            // auto num_key = gen_num_key(it);
            // auto value = num_lookup[num_key];
            // std::cout << key_str << " " << num_key << " " << value << std::endl;
            // ++it;


            // Check for negative numbers
            // int* s_lookup = sign_table[*it];
            // it += sign_adv_table[*it];

            // while (*it != '\n') {
            //   // Skip the decimal.
            //   if (*it == '.') {
            //     ++it;
            //     continue;
            //   }
            //   value = value * 10 + (*it - '0');
              // ++it;
            // }
            // ++it;

            // Update the result map
            // result[key].update(key_str, value);

            // ++inner_counter;
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
        thread_results[i].reserve(1024 << 3);
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
          auto& cr = combinedResults[kv.second.name];
          if (kv.second.min < cr.min) cr.min = kv.second.min;
          if (kv.second.max > cr.max) cr.max = kv.second.max;
          cr.sum += kv.second.sum;
          cr.count += kv.second.count;
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

    return 0;
}
