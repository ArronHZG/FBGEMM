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
  static constexpr size_t kDimension = 128;  // embedding维度
  using scalar_t = float;                    // 数据类型

  void SetUp() override {
    block_size_ = kv_mem::FixedBlockPool::calculate_block_size<scalar_t>(kDimension);
    block_alignment_ = kv_mem::FixedBlockPool::calculate_block_alignment<scalar_t>();
    pool_ = std::make_unique<kv_mem::FixedBlockPool>(block_size_, block_alignment_);
  }

  // 生成随机数据
  void generateRandomData(std::size_t num_blocks) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> key_dist(1, UINT64_MAX);
    std::uniform_real_distribution<float> val_dist(-1.0, 1.0);

    for (size_t i = 0; i < num_blocks; ++i) {
      auto* block = pool_->allocate_t<scalar_t>();
      uint64_t key = key_dist(gen);

      // 设置元数据
      kv_mem::FixedBlockPool::set_key(block, key);
      kv_mem::FixedBlockPool::set_count(block, i % 100);
      kv_mem::FixedBlockPool::update_timestamp(block);

      // 设置embedding数据
      auto* data = kv_mem::FixedBlockPool::data_ptr(block);
      for (size_t j = 0; j < kDimension; ++j) {
        data[j] = val_dist(gen);
      }

      // 记录用于验证
      original_data_[key] = std::vector<scalar_t>(data, data + kDimension);
    }
  }

  // 验证数据正确性
  bool verifyData() {
    size_t verified_count = 0;

    // 遍历所有chunks验证数据
    for (const auto& chunk : pool_->get_chunks()) {
      char* current = static_cast<char*>(chunk.ptr);
      size_t blocks_in_chunk = chunk.size / block_size_;

      for (size_t i = 0; i < blocks_in_chunk; ++i) {
        void* block = current + i * block_size_;
        if (kv_mem::FixedBlockPool::get_used(block)) {
          uint64_t key = kv_mem::FixedBlockPool::get_key(block);
          auto* data = kv_mem::FixedBlockPool::data_ptr(reinterpret_cast<scalar_t*>(block));

          // 查找并比较原始数据
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

  // 性能测试辅助函数
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

// 正确性测试
TEST_F(FixedBlockPoolTest, SerializationCorrectness) {
  // 1. 生成随机数据
  generateRandomData(1000);

  // 2. 序列化
  const std::string filename = "test_pool.bin";
  pool_->serialize(filename);

  // 3. 创建新的内存池并反序列化
  auto new_pool = std::make_unique<kv_mem::FixedBlockPool>(block_size_, block_alignment_);
  new_pool->deserialize(filename);

  // 4. 验证数据
  pool_ = std::move(new_pool);
  EXPECT_TRUE(verifyData());
}

// 边界条件测试
TEST_F(FixedBlockPoolTest, SerializationEdgeCases) {
  // 1. 空池序列化测试
  const std::string empty_filename = "empty_pool.bin";
  pool_->serialize(empty_filename);

  auto new_pool = std::make_unique<kv_mem::FixedBlockPool>(block_size_, block_alignment_);
  EXPECT_NO_THROW(new_pool->deserialize(empty_filename));

  // 2. 文件不存在测试
  EXPECT_THROW(pool_->deserialize("nonexistent_file.bin"), std::runtime_error);

  // 3. 参数不匹配测试
  generateRandomData(1000);
  const std::string filename = "test_pool.bin";
  pool_->serialize(filename);

  auto wrong_pool = std::make_unique<kv_mem::FixedBlockPool>(block_size_ * 2,  // 错误的block大小
                                                             block_alignment_);
  EXPECT_THROW(wrong_pool->deserialize(filename), std::invalid_argument);
}

// 性能测试
TEST_F(FixedBlockPoolTest, SerializationPerformance) {
  // 1. 生成随机数据
  const std::size_t num_blocks = 20'000'000;
  generateRandomData(num_blocks);
  const std::string filename = "test_pool.bin";
  removeFileIfExists(filename);

  // 2. 测试序列化性能
  double serialize_time = measureTime([&]() { pool_->serialize(filename); });

  // 3. 测试反序列化性能
  double deserialize_time = measureTime([&]() {
    auto new_pool = std::make_unique<kv_mem::FixedBlockPool>(block_size_, block_alignment_);
    new_pool->deserialize(filename);
  });

  // 4. 输出性能数据
  double data_size_mb = static_cast<double>((block_size_ * num_blocks)) / (1024.0 * 1024.0);
  std::cout << "\nPerformance Results:" << std::endl;
  std::cout << "Data size: " << data_size_mb << " MB" << std::endl;
  std::cout << "Serialization time: " << serialize_time << " seconds" << std::endl;
  std::cout << "Serialization throughput: " << (data_size_mb / serialize_time) << " MB/s" << std::endl;
  std::cout << "Deserialization time: " << deserialize_time << " seconds" << std::endl;
  std::cout << "Deserialization throughput: " << (data_size_mb / deserialize_time) << " MB/s" << std::endl;
}

}  // namespace kv_mem