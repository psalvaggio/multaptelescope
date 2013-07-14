project (MATS)

cmake_minimum_required (VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MATS_SOURCE_DIR}/cmake)

set(LIB_MACROS "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} std=c++11")

# Find external libraries
find_package(OpenCV REQUIRED)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(    NOT cmake_build_type_tolower STREQUAL "debug"
   AND NOT cmake_build_type_tolower STREQUAL "release")
  message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release (case-insensitive).")
endif()

include_directories(${MATS_SOURCE_DIR})
include_directories(${MATS_BINARY_DIR})

set(CMAKE_CXX_FLAGS "-Wall")
if (cmake_build_type_tolower STREQUAL "release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -force_load")

set(MODULES base
            io
            optical_designs
            third_party)
set(MATS_LINK_LIBS "")
foreach(TMP_LIB ${MODULES})
  add_subdirectory(${TMP_LIB})
endforeach(TMP_LIB)

add_executable(mats_main mats_main.cpp)
target_link_libraries(mats_main ${MATS_LINK_LIBS})

add_executable(write_my_detector write_my_detector.cpp)
target_link_libraries(write_my_detector ${MATS_LINK_LIBS})
