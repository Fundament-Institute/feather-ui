cmake_minimum_required(VERSION 3.13.4)
project(fgGLFW LANGUAGES C CXX VERSION 0.1.0)

find_package(glfw3 CONFIG REQUIRED)

if(MSVC)
  string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
  string(REPLACE "-fexceptions" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

file(GLOB_RECURSE fgGLFW_SOURCES "./*.cpp")

if(MSVC)
file(GLOB_RECURSE fgGLFW_HEADERS "./*.hpp")
add_library(fgGLFW SHARED ${fgGLFW_SOURCES} ${fgGLFW_HEADERS})
  target_link_options(fgGLFW PRIVATE "$<$<CONFIG:Release>:/LTCG>")
else()
add_library(fgGLFW SHARED ${fgGLFW_SOURCES})
endif()

set_property(TARGET fgGLFW PROPERTY C_STANDARD 17)
set_property(TARGET fgGLFW PROPERTY CXX_STANDARD 20)
set_property(TARGET fgGLFW PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET fgGLFW PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET fgGLFW PROPERTY VERBOSE_MAKEFILE TRUE)

retarget_output(fgGLFW)
target_include_directories(fgGLFW PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(fgGLFW PUBLIC ${PROJECT_SOURCE_DIR})

if(MSVC)
  target_compile_options(fgGLFW PRIVATE /Zc:preprocessor $<$<CONFIG:Release>:/Oi /Ot /GL> ${CPP_WARNINGS})
else()
  target_compile_options(fgGLFW PRIVATE -pedantic -fno-exceptions -fno-rtti $<IF:$<CONFIG:Debug>,-g -msse -msse2 -O0,-O3 -msse -msse2 -msse3 -mmmx -m3dnow -mcx16> ${CPP_WARNINGS})
  target_compile_definitions(fgGLFW PUBLIC $<IF:$<CONFIG:Debug>,DEBUG,NDEBUG>)
endif()

if(WIN32)
  find_package(mimalloc CONFIG REQUIRED)
endif()

if(WIN32)
  target_link_libraries(fgGLFW PRIVATE glfw mimalloc-static winmm.lib)
else()
  target_link_libraries(fgGLFW PRIVATE ${CMAKE_DL_LIBS} glfw)
endif()