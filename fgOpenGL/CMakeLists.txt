cmake_minimum_required(VERSION 3.13.4)
project(fgOpenGL LANGUAGES C CXX VERSION 0.1.0)
option(DYNAMIC_RUNTIME "if true, dynamically links (/MD) to the C++ runtime on MSVC. Otherwise, statically links (/MT)" OFF)
option(BUILD_SHARED_LIBS "enable shared library" ON)

find_package(glfw3 REQUIRED)
find_package(Freetype REQUIRED)
find_package(harfbuzz REQUIRED)
find_package(SOIL REQUIRED)
find_package(OpenGL REQUIRED)

if(NOT WIN32)
  find_package(FindFontconfig REQUIRED)
  if(NOT Fontconfig_FOUND)
    message(FATAL_ERROR "Can't find fontconfig! Required to make fonts work on linux.")
  endif()
endif()

if(MSVC)
  set(RUNTIME_FLAG "MT")
  if(DYNAMIC_RUNTIME)
    set(RUNTIME_FLAG "MD")
  endif()
else()
  set(CPP_WARNINGS "-Wall -Wno-attributes -Wno-unknown-pragmas -Wno-missing-braces -Wno-unused-function -Wno-comment -Wno-char-subscripts -Wno-sign-compare -Wno-unused-variable -Wno-switch -Wno-parentheses")
endif()

if(USE32bit)
  set(BIN_DIR "bin-x86")
else()
  set(BIN_DIR "bin-x64")
endif()

set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF) # we use these for (&float)[4][4]

file(GLOB_RECURSE fgOpenGL_SOURCES "./*.cpp")
list(APPEND fgOpenGL_SOURCES "glad_gl.c")

add_library(fgOpenGL ${fgOpenGL_SOURCES})
set_target_properties(fgOpenGL PROPERTIES OUTPUT_NAME_DEBUG "fgOpenGL-d")
target_include_directories(fgOpenGL PUBLIC ${PROJECT_SOURCE_DIR}/../include)

if(WIN32)
target_include_directories(fgOpenGL PRIVATE ${OPENGL_INCLUDE_DIRS} ${FREETYPE_INCLUDE_DIRS} ${SOIL_INCLUDE_DIRS})
else()
target_include_directories(fgOpenGL PRIVATE ${Fontconfig_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${FREETYPE_INCLUDE_DIRS} ${SOIL_INCLUDE_DIRS})
endif()

# May not be necessary if compiling with nix 
set_target_properties(fgOpenGL
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

if(WIN32)
  target_link_libraries(fgOpenGL PRIVATE ${OPENGL_LIBRARIES} glfw ${FREETYPE_LIBRARIES} harfbuzz ${SOIL_LIBRARIES} )
else()
  target_link_libraries(fgOpenGL PRIVATE ${Fontconfig_LIBRARIES} ${OPENGL_LIBRARIES} glfw ${FREETYPE_LIBRARIES} harfbuzz ${SOIL_LIBRARIES} )
endif()