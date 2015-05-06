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

set (unittest_SRC unittest.cpp)
add_executable (unittest ${unittest_SRC})
target_link_libraries (unittest gtest pthread)
set_target_properties(unittest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/unittest/)
