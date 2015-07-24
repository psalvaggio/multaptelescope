find_package(HDF5 REQUIRED)

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

set(LINK_LIBS hdf5 hdf5_hl)

if (APPLE)
  set(LINK_LIBS ${LINK_LIBS} "-framework SBIGUDrv")
else()
  set(LINK_LIBS ${LINK_LIBS} "-lsbigudrv")
endif()

set(MATS_LINK_LIBS ${MATS_LINK_LIBS} ${LINK_LIBS} PARENT_SCOPE)
