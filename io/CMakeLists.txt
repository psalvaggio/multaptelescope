set(MATS_HDRS ${MATS_HDRS}
    logging.h
    detector_writer.h
    detector_reader.h
    envi_image_reader.h
    envi_image_writer.h
    hdf5_reader.h
    input_reader.h
    sbig_detector.h
    stl_writer.h
    text_file_reader.h
    PARENT_SCOPE)

set(MATS_SRCS ${MATS_SRCS}
    logging.cpp
    detector_writer.cpp
    detector_reader.cpp
    envi_image_reader.cpp
    envi_image_writer.cpp
    hdf5_reader.cpp
    input_reader.cpp
    sbig_detector.cpp
    stl_writer.cpp
    text_file_reader.cpp
    PARENT_SCOPE)
