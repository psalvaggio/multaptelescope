set(MATS_HDRS ${MATS_HDRS}
    command_history.h
    detector_writer.h
    envi_image_reader.h
    envi_image_writer.h
    hdf5_reader.h
    logging.h
    protobuf_reader.h
    sbig_detector.h
    shell.h
    shell.hpp
    stl_writer.h
    text_file_reader.h
    PARENT_SCOPE)

set(MATS_SRCS ${MATS_SRCS}
    command_history.cpp
    detector_writer.cpp
    envi_image_reader.cpp
    envi_image_writer.cpp
    hdf5_reader.cpp
    logging.cpp
    sbig_detector.cpp
    stl_writer.cpp
    text_file_reader.cpp
    PARENT_SCOPE)
