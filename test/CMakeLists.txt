find_package(Threads REQUIRED)
find_threads_cuda_fix()

add_library(gtest EXCLUDE_FROM_ALL STATIC gtest-all.cpp)
target_include_directories(gtest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gtest PUBLIC Threads::Threads)

# Convenience target that builds all tests.
# Test executable targets should be added to the 'tests' target as dependencies.
add_custom_target(tests)

# Unit tests.
# Builds: unit.
add_subdirectory(unit)
