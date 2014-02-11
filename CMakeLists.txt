project (MATS)

cmake_minimum_required (VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MATS_SOURCE_DIR}/cmake)

set(LIB_MACROS "")

# Find external libraries
find_package(OpenCV)
include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

find_package(Boost REQUIRED COMPONENTS iostreams system)

set(PROTOBUF_IMPORT_DIRS ${PROTOBUF_IMPORT_DIRS} ${MATS_SOURCE_DIR})

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
if (cmake_build_type_tolower STREQUAL "release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()


ADD_CUSTOM_COMMAND(
    OUTPUT SOMEFILE.pb.h SOMEFILE.pb.cc
    COMMAND protoc SOMEFILE.proto --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
    SOMEFILE.proto
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})


set(MODULES base
            deconvolution
            io
            optical_designs
            otf_measurement
            ransac)
set(MATS_LINK_LIBS "")
foreach(TMP_LIB ${MODULES})
  add_subdirectory(${TMP_LIB})
endforeach(TMP_LIB)

set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${Boost_LIBRARIES})

add_executable(mats_main mats_main.cpp)
target_link_libraries(mats_main ${MATS_LINK_LIBS})

add_executable(create_mask_punchout create_mask_punchout.cpp)
target_link_libraries(create_mask_punchout ${MATS_LINK_LIBS})

add_executable(slant_edge_test slant_edge_test.cpp)
target_link_libraries(slant_edge_test ${MATS_LINK_LIBS})
