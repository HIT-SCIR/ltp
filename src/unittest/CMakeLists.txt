# three library should be include
# - local directory
# - directory where include gtest header
# - project output header
include_directories (./
    ${SOURCE_DIR}/
    ${THIRDPARTY_DIR}/gtest/include
    ${THIRDPARTY_DIR}/boost/include
    ${THIRDPARTY_DIR}/eigen-3.2.4/)
link_directories (${LIBRARY_OUTPUT_PATH})

# add unittest executable
set (unittest_SRC unittest.cpp
    ${SOURCE_DIR}/segmentor/preprocessor.cpp
    ${SOURCE_DIR}/segmentor/decoder.cpp
    ${SOURCE_DIR}/parser.n/io.cpp
    ${SOURCE_DIR}/parser.n/instance.cpp
    ${SOURCE_DIR}/parser.n/options.cpp)
add_executable (unittest ${unittest_SRC})
target_link_libraries (unittest gtest boost_regex_static_lib pthread)
set_target_properties(unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (model_validation_SRC ./model_validation.cpp)
add_executable (model_validation ${model_validation_SRC})
target_link_libraries (model_validation
    segmentor_static_lib
    postagger_static_lib
    ner_static_lib
    parser_static_lib
    boost_program_options_static_lib)
set_target_properties(model_validation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)
