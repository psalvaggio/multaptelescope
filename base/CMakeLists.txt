set(SUBSYS_NAME base)

find_package(FFTW REQUIRED)

set(headers
    detector.h
    pupil_function.h
    opencv_utils.h
    registry.h
    macros.h
    assertions.h
    str_utils.h
    math_utils.h
    endian.h
    scoped_ptr.h
    system_otf.h
    telescope.h)

set(sources
    detector.cpp
    pupil_function.cpp
    opencv_utils.cpp
    circshift.cpp
    system_otf.cpp
    telescope.cpp)

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
