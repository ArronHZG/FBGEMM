find_package(folly REQUIRED)
find_package(gflags REQUIRED)

include_directories(
        ${FBGEMM_SOURCE_DIR}
)

set(COMMON_COMPILE_FEATURES cxx_std_17)
set(COMMON_COMPILE_OPTIONS "-O3")
set(COMMON_LINK_LIBRARIES gtest gtest_main Folly::folly)

set(TEST_TARGETS
        fixed_block_pool_test
        fixed_block_pool_saver_test
        sharded_map_test
        feature_evict_test
)

foreach(target ${TEST_TARGETS})
    add_executable(${target} ${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp)
    target_compile_features(${target} PUBLIC ${COMMON_COMPILE_FEATURES})
    target_compile_options(${target} PUBLIC ${COMMON_COMPILE_OPTIONS})
    target_link_libraries(${target} ${COMMON_LINK_LIBRARIES})
endforeach()