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
set (utils_strutils_unittest_SRC ./utils_strutils_unittest.cpp)
add_executable (utils_strutils_unittest ${utils_strutils_unittest_SRC})
target_link_libraries (utils_strutils_unittest gtest pthread)
set_target_properties(utils_strutils_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (utils_tinybitset_unittest_SRC ./utils_tinybitset_unittest.cpp)
add_executable (utils_tinybitset_unittest ${utils_tinybitset_unittest_SRC})
target_link_libraries (utils_tinybitset_unittest gtest pthread)
set_target_properties(utils_tinybitset_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (utils_chartypes_unittest_SRC ./utils_chartypes_unittest.cpp)
add_executable (utils_chartypes_unittest ${utils_chartypes_unittest_SRC})
target_link_libraries (utils_chartypes_unittest gtest pthread)
set_target_properties(utils_chartypes_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (utils_codecs_unittest_SRC ./utils_codecs_unittest.cpp)
add_executable (utils_codecs_unittest ${utils_codecs_unittest_SRC})
target_link_libraries (utils_codecs_unittest gtest pthread)
set_target_properties(utils_codecs_unittest PROPERTIES
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

set (engpattern_unittest_SRC ./engpattern_unittest.cpp)
add_executable (engpattern_unittest ${engpattern_unittest_SRC})
target_link_libraries (engpattern_unittest boost_regex gtest pthread)
set_target_properties(engpattern_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)


set (segmentor_unittest_SRC ./segmentor_unittest.cpp)
add_executable (segmentor_unittest ${segmentor_unittest_SRC})
target_link_libraries (segmentor_unittest segmentor gtest pthread)
set_target_properties(segmentor_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (postagger_internal_unittest_SRC ./postagger_internal_unittest.cpp)
add_executable (postagger_internal_unittest
${postagger_internal_unittest_SRC})
target_link_libraries (postagger_internal_unittest postagger gtest pthread)
set_target_properties(postagger_internal_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (postagger_interface_unittest_SRC ./postagger_interface_unittest.cpp)
add_executable (postagger_interface_unittest
${postagger_interface_unittest_SRC})
target_link_libraries (postagger_interface_unittest postagger gtest pthread)
set_target_properties(postagger_interface_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (parser_unittest_SRC ./parser_unittest.cpp)
add_executable (parser_unittest ${parser_unittest_SRC})
target_link_libraries (parser_unittest parser gtest pthread)
set_target_properties(parser_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (parser_interface_unittest_SRC ./parser_interface_unittest.cpp)
add_executable (parser_interface_unittest
${parser_interface_unittest_SRC})
target_link_libraries (parser_interface_unittest parser gtest pthread)
set_target_properties(parser_interface_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)

set (srl_interface_unittest_SRC ./srl_interface_unittest.cpp)
add_executable (srl_interface_unittest
${srl_interface_unittest_SRC})
target_link_libraries (srl_interface_unittest srl gtest pthread)
set_target_properties(srl_interface_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)


