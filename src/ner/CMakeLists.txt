include_directories (${SOURCE_DIR}/)

set (ner_VERSION "0.0.1")

set (ner_SRC
    options.cpp
    decoder.cpp
    featurespace.cpp
    model.cpp
    extractor.cpp
    ner.cpp)

add_library (ner_static_lib STATIC ner_dll.cpp ${ner_SRC})

set_target_properties (ner_static_lib PROPERTIES OUTPUT_NAME ner)

add_library (ner_shared_lib SHARED ner_dll.cpp ${ner_SRC})

set_target_properties (ner_shared_lib PROPERTIES
    VERSION     ${ner_VERSION}
    OUTPUT_NAME ner)

add_executable (otner otner.cpp ${ner_SRC})
#target_link_libraries (otner boost_regex)

# redirect the output binary to tools/train
set_target_properties (otner
    PROPERTIES
    OUTPUT_NAME otner
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)
configure_file (
    ner_dll.h
    ${INCLUDE_OUTPUT_PATH}/ner_dll.h)
