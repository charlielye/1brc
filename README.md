# 1️⃣🐝🏎️ The One Billion Row Challenge

A [C++ implementation](./main.cpp) of the orignal java challenge here: https://github.com/gunnarmorling/1brc

Discussion thread here: https://github.com/gunnarmorling/1brc/discussions/495

## Example usage.

```
clang++-16 -O3 -std=c++20 -march=native -pthread main.cpp
THREADS=8 ./a.out
```

`g++` should also work. You can also specify the number of "linear chunks" with CHUNKS env var. Unmaps will occur at end of each chunk. Default is 8.

```
CHUNKS=8 THREADS=8 ./a.out
```
