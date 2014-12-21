set(SUBSYS_NAME aperture_optimization)
set(SUBSYS_DEPS base)

set(LIB_NAME mats_${SUBSYS_NAME})

set(headers
    golay_genetic_impl.h)

set(sources
    golay_genetic_impl.cpp)

foreach(TMPLIB ${SUBSYS_DEPS})
  set(LINK_LIBS ${LINK_LIBS} mats_${TMPLIB})
endforeach()

add_library(${LIB_NAME} ${headers} ${sources})
target_link_libraries(${LIB_NAME} ${LINK_LIBS} ${OpenCV_Libs})

IF (APPLE)
  set(LIB_NAME -Wl,-force_load ${LIB_NAME})
else()
  set(LIB_NAME -Wl,--whole-archive ${LIB_NAME} -Wl,--no-whole-archive)
endif()

set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LIB_NAME} PARENT_SCOPE)
