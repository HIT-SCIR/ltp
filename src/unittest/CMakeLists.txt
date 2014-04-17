# three library should be include
# - local directory
# - directory where include gtest header
# - project output header
include_directories (./
    ${SOURCE_DIR}/
    ${THIRDPARTY_DIR}/gtest/include)

link_directories (${LIBRARY_OUTPUT_PATH})

# add unittest executable
set (utils_strutils_unittest_SRC ./utils_strutils_unittest.cpp)
add_executable (utils_strutils_unittest ${utils_strutils_unittest_SRC})
target_link_libraries (utils_strutils_unittest gtest pthread)
set_target_properties(utils_strutils_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (utils_template_unittest_SRC ./utils_template_unittest.cpp)
add_executable (utils_template_unittest ${utils_template_unittest_SRC})
target_link_libraries (utils_template_unittest gtest pthread)
set_target_properties(utils_template_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)


set (utils_strpaste_unittest_SRC ./utils_strpaste_unittest.cpp)
add_executable (utils_strpaste_unittest ${utils_strpaste_unittest_SRC})
target_link_libraries (utils_strpaste_unittest gtest pthread)
set_target_properties(utils_strpaste_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (segmentor_unittest_SRC ./segmentor_unittest.cpp)
add_executable (segmentor_unittest ${segmentor_unittest_SRC})
target_link_libraries (segmentor_unittest segmentor gtest pthread)
set_target_properties(segmentor_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (postagger_unittest_SRC ./postagger_unittest.cpp)
add_executable (postagger_unittest ${postagger_unittest_SRC})
target_link_libraries (postagger_unittest postagger gtest pthread)
set_target_properties(postagger_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (parser_unittest_SRC ./parser_unittest.cpp)
add_executable (parser_unittest ${parser_unittest_SRC})
target_link_libraries (parser_unittest parser gtest pthread)
set_target_properties(parser_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)
