set(SUBSYS_NAME ransac)

set(headers
    ransac.h
    ransac_fit_circle.h
    ransac_fit_line.h)

set(sources
    ransac.cpp
    ransac_fit_circle.cpp
    ransac_fit_line.cpp)

set(LIB_NAME mats_${SUBSYS_NAME})
set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LIB_NAME} PARENT_SCOPE)

add_library(${LIB_NAME} ${headers} ${sources})
target_link_libraries(${LIB_NAME} ${OpenCV_LIBS})
