set(SUBSYS_NAME optical_designs)
set(SUBSYS_DEPS base io)

set(headers
    aperture.h
    cassegrain.h)

set(sources
    aperture.cpp
    cassegrain.cpp)

set(LIB_NAME mats_${SUBSYS_NAME})

foreach(TMPLIB ${SUBSYS_DEPS})
  set(LINK_LIBS ${LINK_LIBS} mats_${TMPLIB})
endforeach()

set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LIB_NAME} PARENT_SCOPE)

add_library(${LIB_NAME} STATIC ${headers} ${sources})
target_link_libraries(${LIB_NAME} ${LINK_LIBS} ${OpenCV_Libs})
