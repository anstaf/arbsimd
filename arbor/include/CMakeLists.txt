add_library(arbor-public-headers INTERFACE)

# At build time, public headers found in this directory.
# For consumers of an installed arbor package, the include files will be found
# in the 'include' subdirectory of the installation path.

target_include_directories(arbor-public-headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(DIRECTORY arbor ${CMAKE_CURRENT_BINARY_DIR}/arbor
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.inc"
    )

install(TARGETS arbor-public-headers EXPORT arbor-targets)
