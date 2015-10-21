project (MATS)

cmake_minimum_required (VERSION 2.6)
set(CMAKE_MACOSX_RPATH 1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MATS_SOURCE_DIR}/cmake)

set(LIB_MACROS "")

# Find external libraries
find_package(OpenCV 3.0.0 REQUIRED)
include(FindProtobuf)
find_package(Protobuf REQUIRED)
find_package(Gflags REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
find_package(Boost REQUIRED COMPONENTS iostreams system filesystem regex)
find_package(FFTW REQUIRED)
find_package(Curses REQUIRED)
find_package(HDF5 REQUIRED)

# Respect environment variables
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CXXFLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} $ENV{LDFLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} $ENV{LDFLAGS}")

# Make sure we have C++11 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11
                  support. Please use a different C++ compiler.")
endif()

# Check the build type
if (NOT (CMAKE_BUILD_TYPE STREQUAL "Release" OR
         CMAKE_BUILD_TYPE STREQUAL "Debug"))
  set(CMAKE_BUILD_TYPE "Release")
endif()
string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(    NOT cmake_build_type_tolower STREQUAL "debug"
   AND NOT cmake_build_type_tolower STREQUAL "release")
  message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release (case-insensitive).")
endif()

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if (cmake_build_type_tolower STREQUAL "release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

if ("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# Allow top-level include paths. Binary directory allows top-level includes to
# also work for protobuf headers.
include_directories(${MATS_SOURCE_DIR})
include_directories(${MATS_BINARY_DIR})

# List of submodules
set(MODULES base
            #aperture_optimization
            deconvolution
            io
            genetic
            optical_designs
            otf_measurement
            ransac)

set(MATS_LINK_LIBS "")
foreach(TMP_LIB ${MODULES})
  # Grab the proto files in the submodule
  unset(ProtoFiles)
  file(GLOB_RECURSE ProtoFiles
       RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
       ${CMAKE_CURRENT_SOURCE_DIR}/${TMP_LIB}/*.proto)
  set(${TMP_LIB}_ProtoFiles ${ProtoFiles} CACHE STRING ${TMP_LIB}_ProtoFiles)
  set(${TMP_LIB}_ProtoSources CACHE STRING ${TMP_LIB}_ProtoFiles)
  set(${TMP_LIB}_ProtoHeaders CACHE STRING ${TMP_LIB}_ProtoHeaders)

  # Generate proto source files
  if (${TMP_LIB}_ProtoFiles)
    foreach (PROTO_FILE ${${TMP_LIB}_ProtoFiles})
      get_filename_component(ABS_FIL ${PROTO_FILE} ABSOLUTE)
      get_filename_component(FIL ${PROTO_FILE} NAME_WE)
      get_filename_component(PROTO_PATH ${PROTO_FILE} PATH)
      add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_PATH}/${FIL}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_PATH}/${FIL}.pb.h"
        COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
        ARGS --cpp_out=${CMAKE_BINARY_DIR}#/${PROTO_PATH}
             --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
             --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_PATH}
             ${ABS_FIL}
        DEPENDS ${ABS_FIL}
        COMMENT "Running C++ protocol buffer compiler on ${PROTO_FILE}"
        VERBATIM)
      list(APPEND ${TMP_LIB}_ProtoSources
          "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_PATH}/${FIL}.pb.cc")
      list(APPEND ${TMP_LIB}_ProtoHeaders
          "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_PATH}/${FIL}.pb.h")
    endforeach()
  endif()
endforeach(TMP_LIB)

# Create a list of all the source and headers. Grab each submodule's proto files
# and include their CMakeLists.txt, which adds their C++ files.
set(SRCS "")
set(HDRS "")
foreach(TMP_LIB ${MODULES})
  if (${TMP_LIB}_ProtoFiles)
    set(SRCS ${SRCS} ${${TMP_LIB}_ProtoSources})
    set(HDRS ${HDRS} ${${TMP_LIB}_ProtoHeaders})
  endif()

  set(MATS_SRCS "")
  set(MATS_HDRS "")
  add_subdirectory(${TMP_LIB})

  foreach(TMP_FILE ${MATS_SRCS})
    list(APPEND SRCS ${TMP_LIB}/${TMP_FILE})
  endforeach()
  foreach(TMP_FILE ${MATS_HDRS})
    list(APPEND HDRS ${TMP_LIB}/${TMP_FILE})
  endforeach()
endforeach(TMP_LIB)

# Aggregate our dependencies
list(APPEND MATS_LINK_LIBS ${Boost_LIBRARIES}
                           ${OpenCV_LIBRARIES}
                           ${PROTOBUF_LIBRARIES}
                           ${GFLAGS_LIBRARIES}
                           ${FFTW_LIBRARIES}
                           ${CURSES_LIBRARIES}
                           "-lmenu"
                           hdf5
                           hdf5_hl
                           )
if (APPLE)
  list(APPEND MATS_LINK_LIBS "-framework SBIGUDrv")
else()
  list(APPEND MATS_LINK_LIBS "-lsbigudrv")
endif()

set (LIB_NAME mats)

# Create a shared library with all of our source files
add_library(${LIB_NAME} SHARED ${SRCS} ${HDRS})
target_link_libraries(${LIB_NAME} ${MATS_LINK_LIBS})

# To make registries work, we need to force all (even unused) symbols to be
# loaded.
if (APPLE)
  set(LIB_NAME -Wl,-force_load ${LIB_NAME})
else()
  set(LIB_NAME -Wl,--whole-archive ${LIB_NAME} -Wl,--no-whole-archive)
endif()

list(APPEND MATS_LINK_LIBS ${LIB_NAME})

# List of executables
set(EXES aberrated_otf
         acquire_images
         align_measured_profiles
         analyze_usaf_1951
         center_mask
         create_spectral_weighting
         #genetic_optimization
         image_binary_target
         mats_main
         mtf_plot
         mtf_statistics
         real_time_sbig_preview
         restore_image
         slant_edge
         #compare_tribar_profiles
         #perform_rer_comparison
         )

# Build each executable
foreach(EXECUTABLE ${EXES})
  add_executable(${EXECUTABLE} ${EXECUTABLE}.cpp)
  target_link_libraries(${EXECUTABLE} ${MATS_LINK_LIBS})
endforeach()
