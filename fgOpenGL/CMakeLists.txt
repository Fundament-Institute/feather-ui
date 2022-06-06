cmake_minimum_required(VERSION 3.13.4)
project(fgOpenGL LANGUAGES C CXX VERSION 0.1.0)
option(DYNAMIC_RUNTIME "if true, dynamically links (/MD) to the C++ runtime on MSVC. Otherwise, statically links (/MT)" OFF)
option(BUILD_SHARED_LIBS "enable shared library" ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

if(MSVC)
  set(RUNTIME_FLAG "MT")
  if(DYNAMIC_RUNTIME)
    set(RUNTIME_FLAG "MD")
  endif()
else()
  set(CPP_WARNINGS "-Wall -Wno-attributes -Wno-unknown-pragmas -Wno-missing-braces -Wno-unused-function -Wno-comment -Wno-char-subscripts -Wno-sign-compare -Wno-unused-variable -Wno-switch -Wno-parentheses")
endif()

set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # we use these for (&float)[4][4]

file(GLOB_RECURSE fgOpenGL_SOURCES "./*.cpp")
list(APPEND fgOpenGL_SOURCES "glad.c")

add_library(fgOpenGL ${fgOpenGL_SOURCES})
retarget_output(fgOpenGL)
target_include_directories(fgOpenGL PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(fgOpenGL PRIVATE ${PROJECT_SOURCE_DIR})

if(WIN32)
target_include_directories(fgOpenGL PRIVATE ${OPENGL_INCLUDE_DIRS} )
else()
target_include_directories(fgOpenGL PRIVATE ${Fontconfig_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})
endif()

if(WIN32)
  target_link_libraries(fgOpenGL PRIVATE ${OPENGL_LIBRARIES} glfw)
else()
  target_link_libraries(fgOpenGL PRIVATE ${Fontconfig_LIBRARIES} ${OPENGL_LIBRARIES} glfw)
endif()
