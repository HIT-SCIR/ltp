include_directories (./
    ${SOURCE_DIR}/segmentor
    ${SOURCE_DIR}/utils
    ${SOURCE_DIR}/utils/math)

set (segment_SRC
    options.cpp
    decoder.cpp
    featurespace.cpp
    model.cpp
    extractor.cpp
    segmentor.cpp
    otcws.cpp)

add_executable (otcws ${segment_SRC})

set (segment_dll_SRC
    segment_dll.cpp
    ${segment_SRC})

add_library (segmentor ${segment_dll_SRC})
