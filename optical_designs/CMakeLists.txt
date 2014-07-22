set(SUBSYS_NAME optical_designs)
set(SUBSYS_DEPS base io base_proto)

set(LIB_NAME mats_${SUBSYS_NAME})

set(headers
    aperture.h
    cassegrain.h
    circular.h
    hdf5_wfe.h
    triarm3.h
    triarm9.h
    two_cassegrain.h)

set(sources
    aperture.cpp
    circular.cpp
    cassegrain.cpp
    hdf5_wfe.cpp
    triarm3.cpp
    triarm9.cpp
    two_cassegrain.cpp)

foreach(TMPLIB ${SUBSYS_DEPS})
  set(LINK_LIBS ${LINK_LIBS} mats_${TMPLIB})
endforeach()

set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LIB_NAME} PARENT_SCOPE)

add_library(${LIB_NAME} ${headers} ${sources})
target_link_libraries(${LIB_NAME} mats_${SUBSYS_NAME}_proto
                                  ${LINK_LIBS} ${OpenCV_Libs})
