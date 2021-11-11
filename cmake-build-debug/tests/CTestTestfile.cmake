# CMake generated Testfile for 
# Source directory: /Users/antonaf/arbor-ws/arbsimd/tests
# Build directory: /Users/antonaf/arbor-ws/arbsimd/cmake-build-debug/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_simd "/Users/antonaf/arbor-ws/arbsimd/cmake-build-debug/tests/test_simd")
set_tests_properties(test_simd PROPERTIES  _BACKTRACE_TRIPLES "/Users/antonaf/arbor-ws/arbsimd/tests/CMakeLists.txt;19;add_test;/Users/antonaf/arbor-ws/arbsimd/tests/CMakeLists.txt;22;arbsimd_add_test;/Users/antonaf/arbor-ws/arbsimd/tests/CMakeLists.txt;0;")
add_test(test_simd_value "/Users/antonaf/arbor-ws/arbsimd/cmake-build-debug/tests/test_simd_value")
set_tests_properties(test_simd_value PROPERTIES  _BACKTRACE_TRIPLES "/Users/antonaf/arbor-ws/arbsimd/tests/CMakeLists.txt;19;add_test;/Users/antonaf/arbor-ws/arbsimd/tests/CMakeLists.txt;23;arbsimd_add_test;/Users/antonaf/arbor-ws/arbsimd/tests/CMakeLists.txt;0;")
subdirs("../_deps/googletest-build")
