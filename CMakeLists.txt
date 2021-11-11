cmake_minimum_required(VERSION 3.14.5)
cmake_policy(VERSION 3.14.5)

include(CMakeDependentOption)

file(READ VERSION FULL_VERSION_STRING)
string(STRIP "${FULL_VERSION_STRING}" FULL_VERSION_STRING)
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)?(\\.[0-9]+)?(\\.[0-9]+)?" numeric_version "${FULL_VERSION_STRING}")

project(arbsimd VERSION ${numeric_version})
enable_language(CXX)

include(GNUInstallDirs)

add_library(arbsimd INTERFACE)
target_include_directories(arbsimd INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
else()
    # User setting ARBSIMD_BUILD_TESTING: by default testing is disabled if included via FetchContent/add_subdirectory,
    # GT_BUILD_TESTING=ON will overwrite this default.
    option(ARBSIMD_BUILD_TESTING "Build ArbSIMD tests" OFF)
    if(ARBSIMD_BUILD_TESTING)
        include(CTest)
    endif()
    # Note that the following is setting a local variable BUILD_TESTING, not affecting the setting of a super-project
    set(BUILD_TESTING ${ARBSIMD_BUILD_TESTING})
endif()

if(BUILD_TESTING)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_subdirectory(tests)
endif()

## Build paths.
#
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#
## Generate a .json file with full compilation command for each file.
#
#set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")
#
## Compiler options common to library, examples, tests, etc.
#
#include("CompilerOptions")
#add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${CXXOPT_WALL}>")
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

#----------------------------------------------------------
# Set up install paths, permissions.
#----------------------------------------------------------

# Set up install paths according to GNU conventions.
#
# GNUInstallDirs picks (e.g.) `lib64` for the library install path on some
# systems where this is definitely not correct (e.g. Arch Linux). If there
# are cases where `lib` is inappropriate, we will have to incorporate special
# case behaviour here.

if(NOT CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
endif()
include(GNUInstallDirs)

# Implicitly created directories require permissions to be set explicitly
# via this CMake variable.
#
# Note that this has no effect until CMake version 3.11.

#set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
#    OWNER_READ
#    OWNER_WRITE
#    OWNER_EXECUTE
#    GROUP_READ
#    GROUP_EXECUTE
#    WORLD_READ
#    WORLD_EXECUTE)

# CMake versions 3.11 and 3.12 ignore this variable for directories
# implicitly created by install(DIRECTORY ...), which for us corresponds
# to our doc and include directories. Work-around by trying to install
# a non-existant file to these locations.

#foreach(directory "${CMAKE_INSTALL_DOCDIR}" "${CMAKE_INSTALL_INCLUDEDIR}")
#    install(FILES _no_such_file_ OPTIONAL DESTINATION "${directory}")
#endforeach()

#----------------------------------------------------------
# Configure targets in sub-directories.
#----------------------------------------------------------

# html:
#add_subdirectory(doc)

#----------------------------------------------------------
# Generate CMake config/version files for install.
#----------------------------------------------------------

# Note: each dependency for the arbor library target, private or otherwise,
# needs to add itself to the arbor-exports EXPORT target in the subdirectory
# in which they are defined, or none of this will work.

#set(cmake_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/arbor")
#install(EXPORT arbor-targets NAMESPACE arbor:: DESTINATION "${cmake_config_dir}")

#include(CMakePackageConfigHelpers)
#write_basic_package_version_file(
#    "${CMAKE_CURRENT_BINARY_DIR}/arbor-config-version.cmake"
#    COMPATIBILITY SameMajorVersion)

# Template file will use contents of arbor_export_dependencies to include the
# required `find_dependency` statements, and arbor_supported_components will
# be used to check feature support.
#
# To avoid CMake users of the installed arbor library conditionally requiring
# that they add CUDA to their project language, explicitly munge the import
# language and library dependencies on the installed target if ARB_WITH_GPU
# is set, via the variables arbor_override_import_lang and arbor_add_import_libs.
# arbor_build_config records our build type in a way compatible with the
# generated export cmake files.

#set(arbor_build_config NOCONFIG)
#if(CMAKE_BUILD_TYPE)
#    string(TOUPPER "${CMAKE_BUILD_TYPE}" arbor_build_config)
#endif()
#
#set(arbor_override_import_lang)
#set(arbor_add_import_libs)
#set(arborenv_add_import_libs)
#
## (We remove old generated one so that the generation happens every time we run cmake.)
#file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/arbor-config.cmake")
#configure_file(
#    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/arbor-config.cmake.in"
#    "${CMAKE_CURRENT_BINARY_DIR}/arbor-config.cmake"
#    @ONLY)
#
#install(
#    FILES
#        "${CMAKE_CURRENT_BINARY_DIR}/arbor-config.cmake"
#        "${CMAKE_CURRENT_BINARY_DIR}/arbor-config-version.cmake"
#        cmake/FindUnwind.cmake
#    DESTINATION "${cmake_config_dir}")

