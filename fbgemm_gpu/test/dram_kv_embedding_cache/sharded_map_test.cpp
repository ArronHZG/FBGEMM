#include <cstdio>
#include <iostream>

#include <array>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include "fbgemm_gpu/src/dram_kv_embedding_cache/SynchronizedShardedMap.h"
#include "fbgemm_gpu/src/dram_kv_embedding_cache/fixed_block_pool.h"

namespace kv_mem {
std::vector<float> generateFixedEmbedding(int dimension) { return std::vector<float>(dimension, 1.0); }

void memPoolEmbedding(int dimension, std::size_t numInserts, std::size_t numLookups) {
  const std::size_t numShards = 1;

  SynchronizedShardedMap<unsigned long, float*> embeddingMap(numShards,
                                                             dimension * sizeof(float),  // block_size
                                                             alignof(float),             // block_alignment
                                                             8192);                      // blocks_per_chunk
  double insertTime, lookupTime;
  {
    std::vector<float> fixedEmbedding = generateFixedEmbedding(dimension);

    auto wlock = embeddingMap.by(0).wlock();
    auto* pool = embeddingMap.pool_by(0);
    std::pmr::polymorphic_allocator<float> alloc(pool);

    auto startInsert = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < numInserts; i++) {
      float* arr = alloc.allocate(dimension);
      std::copy(fixedEmbedding.begin(), fixedEmbedding.end(), arr);
      wlock->insert_or_assign(i, arr);
    }
    auto endInsert = std::chrono::high_resolution_clock::now();
    insertTime = std::chrono::duration<double, std::milli>(endInsert - startInsert).count();
  }

  std::vector<float> lookEmbedding(dimension);
  std::size_t hitCount = 0;
  {
    auto rlock = embeddingMap.by(0).rlock();
    auto startLookup = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < numLookups; i++) {
      auto it = rlock->find(i % numInserts);
      if (it != rlock->end()) {
        hitCount++;
        std::copy(it->second, it->second + dimension, lookEmbedding.data());
      }
    }
    auto endLookup = std::chrono::high_resolution_clock::now();
    lookupTime = std::chrono::duration<double, std::milli>(endLookup - startLookup).count();
  }

  fmt::print("{:<20}{:<20.2f}{:<20.2f}{:<20.2f}\n",
             dimension,
             insertTime,
             lookupTime,
             100.0 * static_cast<double>(hitCount) / static_cast<double>(numLookups));
}

void memPoolEmbeddingWithTime(int dimension, std::size_t numInserts, std::size_t numLookups) {
  const std::size_t numShards = 1;
  std::size_t block_size = FixedBlockPool::calculate_block_size<float>(dimension);
  std::size_t block_alignment = FixedBlockPool::calculate_block_alignment<float>();

  SynchronizedShardedMap<unsigned long, float*> embeddingMap(numShards,
                                                             block_size,       // block_size
                                                             block_alignment,  // block_alignment
                                                             8192);            // blocks_per_chunk
  double insertTime, lookupTime;
  {
    std::vector<float> fixedEmbedding = generateFixedEmbedding(dimension);

    auto wlock = embeddingMap.by(0).wlock();
    auto* pool = embeddingMap.pool_by(0);

    auto startInsert = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < numInserts; i++) {
      auto* block = pool->allocate_t<float>();
      auto* data_ptr = FixedBlockPool::data_ptr<float>(block);
      std::copy(fixedEmbedding.begin(), fixedEmbedding.end(), data_ptr);
      wlock->insert_or_assign(i, block);
    }
    auto endInsert = std::chrono::high_resolution_clock::now();
    insertTime = std::chrono::duration<double, std::milli>(endInsert - startInsert).count();
  }

  std::vector<float> lookEmbedding(dimension);
  std::size_t hitCount = 0;
  {
    auto rlock = embeddingMap.by(0).rlock();
    auto startLookup = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < numLookups; i++) {
      auto it = rlock->find(i % numInserts);
      if (it != rlock->end()) {
        hitCount++;
        const float* data_ptr = FixedBlockPool::data_ptr<float>(it->second);
        // update timestamp
        FixedBlockPool::update_timestamp(it->second);
        std::copy(data_ptr, data_ptr + dimension, lookEmbedding.data());
      }
    }
    auto endLookup = std::chrono::high_resolution_clock::now();
    lookupTime = std::chrono::duration<double, std::milli>(endLookup - startLookup).count();
  }

  // 替换输出部分
  fmt::print("{:<20}{:<20.2f}{:<20.2f}{:<20.2f}\n",
             dimension,
             insertTime,
             lookupTime,
             100.0 * static_cast<double>(hitCount) / static_cast<double>(numLookups));
}

int benchmark() {
  std::vector<int> dimensions = {4, 8, 16, 32, 64};
  const std::size_t numInserts = 1'000'000;  // 1 million insert
  const std::size_t numLookups = 1'000'000;  // 1 million find

  fmt::print("======================= mempool ====================================\n");
  fmt::print("{:<20}{:<20}{:<20}{:<20}\n", "dim", "insert time (ms)", "find time (ms)", "hit rate (%)");
  for (int dim : dimensions) {
    memPoolEmbedding(dim, numInserts, numLookups);
  }
  fmt::print("\n\n");
  std::fflush(stdout);

  fmt::print("======================= mempool with time ====================================\n");
  fmt::print("{:<20}{:<20}{:<20}{:<20}\n", "dim", "insert time (ms)", "find time (ms)", "hit rate (%)");
  for (int dim : dimensions) {
    memPoolEmbeddingWithTime(dim, numInserts, numLookups);
  }
  fmt::print("\n\n");
  return 0;
}

void save_and_restore() {
  const int numShards = 4;
  const std::size_t dimension = 32;
  const std::size_t block_size = FixedBlockPool::calculate_block_size<float>(dimension);
  const std::size_t block_alignment = FixedBlockPool::calculate_block_alignment<float>();
  const int numItems = 1'000'000;
  const std::string filename = "test_map.bin";

  SynchronizedShardedMap<int64_t, float*> original_map(numShards, block_size, block_alignment);

  std::vector<float> test_embedding = generateFixedEmbedding(dimension);
  for (int i = 0; i < numItems; ++i) {
    int shard_id = i % numShards;
    auto wlock = original_map.by(shard_id).wlock();
    auto* pool = original_map.pool_by(shard_id);

    auto* block = pool->allocate_t<float>();
    auto* data_ptr = FixedBlockPool::data_ptr<float>(block);
    std::copy(test_embedding.begin(), test_embedding.end(), data_ptr);

    FixedBlockPool::set_key(block, i);
    wlock->insert({i, block});
  }

  original_map.save(filename);

  SynchronizedShardedMap<int64_t, float*> restored_map(numShards, block_size, block_alignment);
  restored_map.load(filename);

  for (int64_t i = 0; i < numItems; ++i) {
    int shard_id = i % numShards;
    auto rlock = restored_map.by(shard_id).rlock();

    auto it = rlock->find(i);
    ASSERT_NE(it, rlock->end()) << "Key " << i << " not found after load";

    float* block = it->second;
    ASSERT_EQ(FixedBlockPool::get_key(block), i);

    const float* data_ptr = FixedBlockPool::data_ptr<float>(block);
    for (std::size_t j = 0; j < dimension; ++j) {
      ASSERT_FLOAT_EQ(data_ptr[j], test_embedding[j]) << "Data mismatch at position " << j << " for key " << i;
    }
  }

  std::remove(filename.c_str());
  for (int i = 0; i < numShards; ++i) {
    std::remove((filename + ".pool." + std::to_string(i)).c_str());
  }
};

TEST(SynchronizedShardedMap, benchmark) { benchmark(); }

TEST(SynchronizedShardedMap, save_and_restore) { save_and_restore(); }

}  // namespace kv_mem