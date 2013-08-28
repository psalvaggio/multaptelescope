set(SUBSYS_NAME deconvolution)

find_package(FFTW REQUIRED)

set(headers
    constrained_least_squares.h)

set(sources
    constrained_least_squares.cpp)

set(LIB_NAME mats_${SUBSYS_NAME})
set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LIB_NAME} PARENT_SCOPE)

add_library(${LIB_NAME} ${headers} ${sources})
target_link_libraries(${LIB_NAME} ${OpenCV_LIBS})
