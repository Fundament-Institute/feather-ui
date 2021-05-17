cmake_minimum_required(VERSION 3.13.4)
project(cpptest LANGUAGES C CXX VERSION 0.1.0)
option(DYNAMIC_RUNTIME "if true, dynamically links (/MD) to the C++ runtime on MSVC. Otherwise, statically links (/MT)" OFF)

find_package(OpenGL REQUIRED)

include(CheckSymbolExists)
include(CMakePushCheckState)

file(GLOB_RECURSE cpptest_SOURCES "./*.cpp")

if(MSVC)
  set(RUNTIME_FLAG "MT")
  if(DYNAMIC_RUNTIME)
    set(RUNTIME_FLAG "MD")
  endif()
else()
  set(CPP_WARNINGS "-Wall -Wno-attributes -Wno-unknown-pragmas -Wno-missing-braces -Wno-unused-function -Wno-comment -Wno-char-subscripts -Wno-sign-compare -Wno-unused-variable -Wno-switch -Wno-parentheses")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

if(MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /${RUNTIME_FLAG}d")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi /Ot /GL /${RUNTIME_FLAG}")
  set(CMAKE_CXX_FLAGS_MINSIZEREL ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELEASE})
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -msse -msse2 -msse3 -mmmx -m3dnow -mcx1 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_MINSIZEREL ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELEASE})
endif()

if(USE32bit)
  set(BIN_DIR "bin-x86")
else()
  set(BIN_DIR "bin-x64")
endif()

set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cpptest ${cpptest_SOURCES})
set_target_properties(cpptest PROPERTIES OUTPUT_NAME_DEBUG "cpptest-d")
target_include_directories(cpptest PUBLIC ${OPENGL_INCLUDE_DIRS})
target_include_directories(cpptest PUBLIC ${PROJECT_SOURCE_DIR}/../include)

# May not be necessary if compiling with nix 
set_target_properties(cpptest
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../${BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../${BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../${BIN_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/../${BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/../${BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/../${BIN_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_SOURCE_DIR}/../${BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_SOURCE_DIR}/../${BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_SOURCE_DIR}/../${BIN_DIR}"
)

if(MSVC)
target_link_libraries(cpptest PRIVATE fgOpenGL ${OPENGL_LIBRARIES})
else()
target_link_libraries(cpptest PRIVATE fgOpenGL stdc++fs ${OPENGL_LIBRARIES})
endif()