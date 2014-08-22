set(SUBSYS_NAME base)
set(SUBSYS_DEPS optical_designs_proto io_proto)

find_package(FFTW REQUIRED)

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(LIB_NAME mats_${SUBSYS_NAME})
set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LIB_NAME} PARENT_SCOPE)

foreach(TMPLIB ${SUBSYS_DEPS})
  set(LINK_LIBS ${LINK_LIBS} mats_${TMPLIB})
endforeach()

find_package(CUDA)
file(GLOB cuda_sources "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
if (${CMAKE_SYSTEM_NAME} MATCHES "Linxu" AND CUDA_FOUND)
  cuda_compile(${LIB_NAME}_cuda_o ${cuda_sources})
  cuda_add_library(${LIB_NAME}_cuda ${${LIB_NAME}_cuda_o})
  set(LINK_LIBS ${LINK_LIBS} ${LIB_NAME}_cuda)
else()
  foreach (f ${cuda_sources})
    set_source_files_properties(${f} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${f} PROPERTIES COMPILE_FLAGS "-x c++")
  endforeach()
  set(sources ${sources} ${cuda_sources})
endif()

add_library(${LIB_NAME} ${headers} ${sources})
target_link_libraries(${LIB_NAME} mats_${SUBSYS_NAME}_proto
                                  ${PROTOBUF_LIBRARY}
                                  ${OpenCV_LIBS}
                                  ${FFTW_LIBRARIES}
                                  ${LINK_LIBS})
