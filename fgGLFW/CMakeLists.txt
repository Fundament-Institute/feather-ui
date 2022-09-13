cmake_minimum_required(VERSION 3.13.4)
project(fgGLFW LANGUAGES C CXX VERSION 0.1.0)

find_package(glfw3 CONFIG REQUIRED)

if(MSVC)
  string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPP_WARNINGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi /Ot /GL")
  set(CMAKE_CXX_FLAGS_MINSIZEREL ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_DEBUG})
else()
  string(REPLACE "-fexceptions" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -fno-exceptions -fno-rtti ${CPP_WARNINGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -msse -msse2 -O0")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -msse -msse2 -msse3 -mmmx -m3dnow -mcx16 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_MINSIZEREL ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_DEBUG})
endif()

set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE fgGLFW_SOURCES "./*.cpp")

if(MSVC)
file(GLOB_RECURSE fgGLFW_HEADERS "./*.hpp")
add_library(fgGLFW SHARED ${fgGLFW_SOURCES} ${fgGLFW_HEADERS})
  target_link_options(fgGLFW PRIVATE "$<$<CONFIG:Release>:/LTCG>")
else()
add_library(fgGLFW SHARED ${fgGLFW_SOURCES})
endif()

retarget_output(fgGLFW)
target_include_directories(fgGLFW PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(fgGLFW PUBLIC ${PROJECT_SOURCE_DIR})


if(WIN32)
  find_package(mimalloc CONFIG REQUIRED)
endif()

if(WIN32)
  target_link_libraries(fgGLFW PRIVATE glfw mimalloc-static winmm.lib)
else()
  target_link_libraries(fgGLFW PRIVATE ${CMAKE_DL_LIBS} glfw)
endif()