# three library should be include
# - local directory
# - directory where include gtest header
# - project output header
include_directories (./
    ${SOURCE_DIR}/
    ${THIRDPARTY_DIR}/gtest/include)

# add unittest executable
#set (postagger_model_unittest_SRC ./postagger_model_test.cpp)
#add_executable (postagger_model_unittest ${postagger_model_unittest_SRC})
#target_link_libraries (postagger_model_unittest gtest pthread)

set (utils_strutils_unittest_SRC ./utils_strutils_unittest.cpp)
add_executable (utils_strutils_unittest ${utils_strutils_unittest_SRC})
target_link_libraries (utils_strutils_unittest gtest pthread)
set_target_properties(utils_strutils_unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)
