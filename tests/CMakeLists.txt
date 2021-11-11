include(FetchContent)
option(INSTALL_GTEST OFF)
mark_as_advanced(INSTALL_GTEST)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)

function(arbsimd_add_test tgt)
    set(options)
    set(one_value_args)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    add_executable(${tgt})
    target_sources(${tgt} PRIVATE ${ARGS_SOURCES})
    target_link_libraries(${tgt} PUBLIC arbsimd gtest_main gmock)
    add_test(NAME ${tgt} COMMAND $<TARGET_FILE:${tgt}>)
endfunction()

arbsimd_add_test(test_simd SOURCES test_simd.cpp)
arbsimd_add_test(test_simd_value SOURCES test_simd_value.cpp)
