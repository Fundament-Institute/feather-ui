cmake_minimum_required(VERSION 3.15)
project(feathergui VERSION 0.1.0)
option(USE_DEFAULT_FOLDERS "Don't override the cmake output folders with a unified /bin/ folder. Also disables debug postfix." OFF)

if(MSVC)
  # This ensures that we default to static but let vcpkg configure things the way it wants
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>$<$<STREQUAL:${VCPKG_CRT_LINKAGE},dynamic>:DLL>" CACHE STRING "")
endif()
set(BIN_DIR "bin")

function(retarget_output target)
  if(NOT USE_DEFAULT_FOLDERS)
    set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "-d")
    set_target_properties(${target}
        PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/${BIN_DIR}"
        PREFIX ""
    )
  endif()
endfunction()

add_subdirectory(fgOpenGL)
#add_subdirectory(fgDirect2D)
add_subdirectory(cpptest)