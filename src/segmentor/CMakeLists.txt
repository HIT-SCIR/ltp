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
    segmentor.cpp)

add_library (segmentor segment_dll.cpp ${segment_SRC})

add_executable (otcws otcws.cpp ${segment_SRC})

# redirect the output binary to tools/train
set_target_properties (otcws
    PROPERTIES
    OUTPUT_NAME otcws
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)


