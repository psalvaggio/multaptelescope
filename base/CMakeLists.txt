set(SUBSYS_NAME base)
set(SUBSYS_DEPS optical_designs_proto)

find_package(FFTW REQUIRED)

set(headers
    assertions.h
    detector.h
    endian.h
    macros.h
    math_utils.h
    mats_init.h
    opencv_utils.h
    photon_noise.h
    pupil_function.h
    registry.h
    scoped_ptr.h
    str_utils.h
    system_otf.h
    telescope.h)

set(sources
    circshift.cpp
    detector.cpp
    mats_init.cpp
    opencv_utils.cpp
    photon_noise.cpp
    pupil_function.cpp
    str_utils.cpp
    system_otf.cpp
    telescope.cpp)

file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")
PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

set(LIB_NAME mats_${SUBSYS_NAME})
set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LIB_NAME} PARENT_SCOPE)

foreach(TMPLIB ${SUBSYS_DEPS})
  set(LINK_LIBS ${LINK_LIBS} mats_${TMPLIB})
endforeach()

add_library(${LIB_NAME} ${ProtoSources}
                        ${ProtoHeaders}
                        ${headers}
                        ${sources})
target_link_libraries(${LIB_NAME} ${PROTOBUF_LIBRARY}
                                  ${OpenCV_LIBS}
                                  ${FFTW_LIBRARIES}
                                  ${LINK_LIBS})

add_executable(registry_test registry_test.cpp registry_test.h registry.h)
