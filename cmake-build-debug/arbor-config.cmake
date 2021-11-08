include(CMakeFindDependencyMacro)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}")

foreach(dep Threads)
    find_dependency(${dep})
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/arbor-targets.cmake")

set(_supported_components )

foreach(component ${arbor_FIND_COMPONENTS})
    if(NOT "${component}" IN_LIST _supported_components)
        set(arbor_FOUND FALSE)
        set(arbor_NOT_FOUND_MESSAGE "Unsupported component: ${component}")
    endif()
endforeach()

# Patch properties to remove unnecessary external CUDA dependencies.

set(_override_lang )
if(_override_lang)
    foreach(target arbor::arbor arbor::arborenv arbor::arborio)
        if(TARGET ${target})
            set_target_properties(${target} PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "${_override_lang}")
        endif()
    endforeach()
endif()

# Explicitly add extra link libraries not covered by dependencies above.
# (See though arbor-sim/arbor issue #678).

function(_append_property target property)
    if (TARGET ${target})
        set(p_append ${ARGN})
        get_target_property(p ${target} ${property})
        if(p)
            list(APPEND p ${p_append})
        else()
            set(interface_libs ${p_append})
        endif()
        set_target_properties(${target} PROPERTIES ${property} "${p}")
    endif()
endfunction()

set(ARB_VECTORIZE OFF)
set(ARB_ARCH native)
set(ARB_MODCC_FLAGS )
set(ARB_CXX /usr/bin/c++)
set(ARB_CXX_FLAGS )
set(ARB_CXX_FLAGS_TARGET -march=native;$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-fvisibility=hidden>;$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fvisibility=hidden>)

_append_property(arbor::arbor INTERFACE_LINK_LIBRARIES )

