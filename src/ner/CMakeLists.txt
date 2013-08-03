include_directories (
    ${SOURCE_DIR}/ner
    ${SOURCE_DIR}/utils
    ${SOURCE_DIR}/utils/math)
#    ${THIRDPARTY_DIR}/boost/include)

set (ner_SRC
    options.cpp
    decoder.cpp
    featurespace.cpp
    model.cpp
    extractor.cpp
    ner.cpp)

add_library (ner ner_dll.cpp ${ner_SRC})

add_executable (otner otner.cpp ${ner_SRC})
link_directories ( ${LIBRARY_OUTPUT_PATH} )
#target_link_libraries (otner boost_regex)

# redirect the output binary to tools/train
set_target_properties (otner
    PROPERTIES
    OUTPUT_NAME otner
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

