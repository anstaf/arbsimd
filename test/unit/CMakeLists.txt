# Unit test sources

set(unit_sources
    test_simd.cpp

    # unit test driver
    test.cpp
)

add_executable(unit EXCLUDE_FROM_ALL ${unit_sources})
add_dependencies(tests unit)

target_compile_options(unit PRIVATE ${ARB_CXX_FLAGS_TARGET_FULL})
target_compile_definitions(unit PRIVATE "-DDATADIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../swc\"")
target_compile_definitions(unit PRIVATE "-DLIBDIR=\"${PROJECT_BINARY_DIR}/lib\"")
target_include_directories(unit PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(unit PRIVATE gtest arbor-public-headers)
