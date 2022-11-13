cmake_minimum_required(VERSION 3.15)
project(backendtest LANGUAGES CXX VERSION 0.1.0)

find_package(OpenGL REQUIRED)

if(MSVC)
  string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
  string(REPLACE "-fexceptions" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

file(GLOB_RECURSE backendtest_SOURCES "./*.c")
add_executable(backendtest ${backendtest_SOURCES})

set_property(TARGET backendtest PROPERTY C_STANDARD 17)
set_property(TARGET backendtest PROPERTY CXX_STANDARD 20)
set_property(TARGET backendtest PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET backendtest PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET backendtest PROPERTY VERBOSE_MAKEFILE TRUE)
set_property(TARGET backendtest PROPERTY POSITION_INDEPENDENT_CODE OFF)

if(MSVC)
  target_compile_options(backendtest PRIVATE /Zc:preprocessor $<$<CONFIG:Release>:/Oi /Ot /GL> ${CPP_WARNINGS})
else()
  target_compile_options(backendtest PRIVATE -pedantic -fno-exceptions -fno-rtti $<IF:$<CONFIG:Debug>,-g -DDEBUG -msse -msse2 -O0,-O3 -msse -msse2 -msse3 -mmmx -m3dnow -mcx16 -DNDEBUG> ${CPP_WARNINGS})
endif()

retarget_output(backendtest)
target_include_directories(backendtest PUBLIC ${OPENGL_INCLUDE_DIRS})
target_include_directories(backendtest PUBLIC ${PROJECT_SOURCE_DIR}/../include)
add_dependencies(backendtest fgOpenGL)

if(WIN32)
  target_link_libraries(backendtest PRIVATE fgOpenGLDesktopBridge fgOpenGL fgGLFW ${OPENGL_LIBRARIES})
  target_link_options(backendtest PRIVATE "$<$<CONFIG:Release>:/LTCG>")
else()
  target_link_libraries(backendtest PRIVATE fgOpenGLDesktopBridge fgOpenGL fgGLFW ${OPENGL_LIBRARIES})
endif()