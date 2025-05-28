#include <chrono>
#include <filesystem>
#include <random>
#include <unordered_map>

#include <gtest/gtest.h>

#include "fbgemm_gpu/src/dram_kv_embedding_cache/fixed_block_pool.h"

namespace kv_mem {
void removeFileIfExists(const std::string& filename) {
  if (std::filesystem::exists(filename)) {
    std::filesystem::remove(filename);
  }
}
class FixedBlockPoolTest : public ::testing::Test {
 protected:
  static constexpr size_t kDimension = 128;  // embedding dimension
  using scalar_t = float;                    // data type

  void SetUp() override {
    block_size_ = kv_mem::FixedBlockPool::calculate_block_size<scalar_t>(kDimension);
    block_alignment_ = kv_mem::FixedBlockPool::calculate_block_alignment<scalar_t>();
    pool_ = std::make_unique<kv_mem::FixedBlockPool>(block_size_, block_alignment_);
  }

  // Generate random data
  void generateRandomData(std::size_t num_blocks) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> key_dist(1, UINT64_MAX);
    std::uniform_real_distribution<float> val_dist(-1.0, 1.0);

    for (size_t i = 0; i < num_blocks; ++i) {
      auto* block = pool_->allocate_t<scalar_t>();
      uint64_t key = key_dist(gen);

      // Set metadata
      kv_mem::FixedBlockPool::set_key(block, key);
      kv_mem::FixedBlockPool::set_count(block, i % 100);
      kv_mem::FixedBlockPool::update_timestamp(block);

      // Set embedding data
      auto* data = kv_mem::FixedBlockPool::data_ptr(block);
      for (size_t j = 0; j < kDimension; ++j) {
        data[j] = val_dist(gen);
      }

      // Record for verification
      original_data_[key] = std::vector<scalar_t>(data, data + kDimension);
    }
  }

  // Verify data correctness
  bool verifyData() {
    size_t verified_count = 0;

    // Traverse all chunks to verify data
    for (const auto& chunk : pool_->get_chunks()) {
      char* current = static_cast<char*>(chunk.ptr);
      size_t blocks_in_chunk = chunk.size / block_size_;

      for (size_t i = 0; i < blocks_in_chunk; ++i) {
        void* block = current + i * block_size_;
        if (kv_mem::FixedBlockPool::get_used(block)) {
          uint64_t key = kv_mem::FixedBlockPool::get_key(block);
          auto* data = kv_mem::FixedBlockPool::data_ptr(reinterpret_cast<scalar_t*>(block));

          // Find and compare original data
          auto it = original_data_.find(key);
          if (it == original_data_.end()) {
            return false;
          }

          if (!std::equal(data, data + kDimension, it->second.begin())) {
            return false;
          }

          verified_count++;
        }
      }
    }

    return verified_count == original_data_.size();
  }

  // Performance test helper function
  template <typename Func>
  double measureTime(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
  }

  std::unique_ptr<kv_mem::FixedBlockPool> pool_;
  size_t block_size_{};
  size_t block_alignment_{};
  std::unordered_map<uint64_t, std::vector<scalar_t>> original_data_;
};

// Correctness test
TEST_F(FixedBlockPoolTest, SerializationCorrectness) {
  // 1. Generate random data
  generateRandomData(1000);

  // 2. Serialize
  const std::string filename = "test_pool.bin";
  pool_->serialize(filename);

  // 3. Create a new memory pool and deserialize
  auto new_pool = std::make_unique<kv_mem::FixedBlockPool>(block_size_, block_alignment_);
  new_pool->deserialize(filename);

  // 4. Verify data
  pool_ = std::move(new_pool);
  EXPECT_TRUE(verifyData());
}

// Edge case test
TEST_F(FixedBlockPoolTest, SerializationEdgeCases) {
  // 1. Empty pool serialization test
  const std::string empty_filename = "empty_pool.bin";
  pool_->serialize(empty_filename);

  auto new_pool = std::make_unique<kv_mem::FixedBlockPool>(block_size_, block_alignment_);
  EXPECT_NO_THROW(new_pool->deserialize(empty_filename));

  // 2. File not found test
  EXPECT_THROW(pool_->deserialize("nonexistent_file.bin"), std::runtime_error);

  // 3. Parameter mismatch test
  generateRandomData(1000);
  const std::string filename = "test_pool.bin";
  pool_->serialize(filename);

  auto wrong_pool = std::make_unique<kv_mem::FixedBlockPool>(block_size_ * 2,  // Incorrect block size
                                                             block_alignment_);
  EXPECT_THROW(wrong_pool->deserialize(filename), std::invalid_argument);
}

// Performance test
TEST_F(FixedBlockPoolTest, SerializationPerformance) {
  const std::size_t num_blocks = 20'000'000;
  generateRandomData(num_blocks);
  const std::string filename = "test_pool.bin";
  removeFileIfExists(filename);

  pool_->serialize(filename);

  auto new_pool = std::make_unique<kv_mem::FixedBlockPool>(block_size_, block_alignment_);
  new_pool->deserialize(filename);

  std::remove(filename.c_str());
}

}  // namespace kv_mem