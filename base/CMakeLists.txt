set(SUBSYS_NAME base)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
find_package(FFTW REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

set(headers
    pupil_function.h
    opencv_utils.h
    registry.h
    macros.h
    assertions.h
    scoped_ptr.h)

set(sources
    pupil_function.cpp
    opencv_utils.cpp
    circshift.cpp)

file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")
PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

set(LIB_NAME mats_${SUBSYS_NAME})
set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LIB_NAME} PARENT_SCOPE)

add_library(${LIB_NAME} ${ProtoSources}
                        ${ProtoHeaders}
                        ${headers}
                        ${sources})
target_link_libraries(${LIB_NAME} ${PROTOBUF_LIBRARY}
                                  ${OpenCV_LIBS}
                                  ${FFTW_LIBRARIES})

add_executable(registry_test registry_test.cpp registry_test.h registry.h)
