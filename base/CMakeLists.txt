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

add_library(${LIB_NAME} ${headers} ${sources})
target_link_libraries(${LIB_NAME} mats_${SUBSYS_NAME}_proto
                                  ${PROTOBUF_LIBRARY}
                                  ${OpenCV_LIBS}
                                  ${FFTW_LIBRARIES}
                                  ${LINK_LIBS})
