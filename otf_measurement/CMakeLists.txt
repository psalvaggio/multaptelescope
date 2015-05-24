set(SUBSYS_NAME otf_measurement)

find_package(FFTW REQUIRED)

set(headers
    hamming_window.h
    jagged_edge_mtf.h
    mtf_interpolator.h
    slant_edge_mtf.h)

set(sources
    hamming_window.cpp
    jagged_edge_mtf.cpp
    mtf_interpolator.cpp
    slant_edge_mtf.cpp)

set(LIB_NAME mats_${SUBSYS_NAME})
set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LIB_NAME} PARENT_SCOPE)

add_library(${LIB_NAME} ${headers} ${sources})
target_link_libraries(${LIB_NAME} ${OpenCV_LIBS} ${FFTW_LIBRARIES})
