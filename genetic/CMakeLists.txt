set(SUBSYS_NAME genetic)

set(headers
    genetic_algorithm.h
    genetic_algorithm.hpp)

set(LIB_NAME mats_${SUBSYS_NAME})

add_custom_target(${LIB_NAME} SOURCES ${headers})
