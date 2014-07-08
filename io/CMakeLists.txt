set(SUBSYS_NAME io)
set(SUBSYS_DEPS base)

set(headers
    logging.h
    detector_writer.h
    detector_reader.h
    envi_image_reader.h
    envi_image_writer.h
    stl_writer.h
    input_reader.h)

set(sources
    logging.cpp
    detector_writer.cpp
    detector_reader.cpp
    envi_image_reader.cpp
    envi_image_writer.cpp
    stl_writer.cpp
    input_reader.cpp)

set(LIB_NAME mats_${SUBSYS_NAME})

foreach(TMPLIB ${SUBSYS_DEPS})
  set(LINK_LIBS ${LINK_LIBS} mats_${TMPLIB})
endforeach()

set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LIB_NAME} PARENT_SCOPE)

add_library(${LIB_NAME} ${headers} ${sources})
target_link_libraries(${LIB_NAME} ${LINK_LIBS} ${LIB_NAME}_proto)
