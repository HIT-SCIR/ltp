# three library should be include
# - local directory
# - directory where include gtest header
# - project output header
include_directories (./
    ${SOURCE_DIR}/
    ${THIRDPARTY_DIR}/gtest/include
    ${THIRDPARTY_DIR}/boost/include)
link_directories (${LIBRARY_OUTPUT_PATH})

# add unittest executable
set (unittest_SRC
    unittest.cpp
    ${SOURCE_DIR}/segmentor/preprocessor.cpp
    ${SOURCE_DIR}/segmentor/decoder.cpp
    ${SOURCE_DIR}/parser.n/io.cpp
    ${SOURCE_DIR}/parser.n/instance.cpp
    ${SOURCE_DIR}/parser.n/options.cpp)
add_executable (unittest ${unittest_SRC})
target_link_libraries (unittest gtest boost_regex_static_lib pthread)
set_target_properties(unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)
