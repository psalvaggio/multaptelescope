find_package(FFTW REQUIRED)

set(MATS_HDRS ${MATS_HDRS}
    hamming_window.h
    jagged_edge_mtf.h
    mtf_interpolator.h
    slant_edge_mtf.h
    PARENT_SCOPE)

set(MATS_SRCS ${MATS_SRCS}
    hamming_window.cpp
    jagged_edge_mtf.cpp
    mtf_interpolator.cpp
    slant_edge_mtf.cpp
    PARENT_SCOPE)

set(MATS_LINK_LIBS ${MATS_LINK_LIBS} 
    ${FFTW_LIBRARIES}
    PARENT_SCOPE)
