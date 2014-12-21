set(SUBSYS_NAME optical_designs)
set(SUBSYS_DEPS base io base_proto)

set(LIB_NAME mats_${SUBSYS_NAME})

set(headers
    aperture.h
    axicon.h
    cassegrain.h
    cassegrain_ring.h
    circular.h
    compound_aperture.h
    hdf5_wfe.h
    optical_vortex.h
    triarm9.h)

set(sources
    aperture.cpp
    axicon.cpp
    cassegrain.cpp
    cassegrain_ring.cpp
    circular.cpp
    compound_aperture.cpp
    hdf5_wfe.cpp
    optical_vortex.cpp
    triarm9.cpp)

foreach(TMPLIB ${SUBSYS_DEPS})
  set(LINK_LIBS ${LINK_LIBS} mats_${TMPLIB})
endforeach()

add_library(${LIB_NAME} ${headers} ${sources})
target_link_libraries(${LIB_NAME} mats_${SUBSYS_NAME}_proto
                                  ${LINK_LIBS} ${OpenCV_Libs})

IF (APPLE)
  set(LIB_NAME -Wl,-force_load ${LIB_NAME})
else()
  set(LIB_NAME -Wl,--whole-archive ${LIB_NAME} -Wl,--no-whole-archive)
endif()

set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LIB_NAME} PARENT_SCOPE)

