cmake_minimum_required (VERSION 2.8.0)
project ("LTP - Language Technology Platform")

# project attributes section
# -- config cmake modules path
set (CMAKE_MODULE_PATH      ${PROJECT_SOURCE_DIR}/cmake)

# -- config output directories
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_PATH    ${PROJECT_SOURCE_DIR}/lib)
set (INCLUDE_OUTPUT_PATH    ${PROJECT_SOURCE_DIR}/include)

# -- config source directories
set (SOURCE_DIR             ${PROJECT_SOURCE_DIR}/src)
set (THIRDPARTY_DIR         ${PROJECT_SOURCE_DIR}/thirdparty)
set (TOOLS_DIR              ${PROJECT_SOURCE_DIR}/tools)

# -- config resource directories
set (CONFIGURE_DIR          ${PROJECT_SOURCE_DIR}/conf)
set (MODEL_DIR              ${PROJECT_SOURCE_DIR}/ltp_data)
set (DATA_DIR               ${PROJECT_SOURCE_DIR}/test_data)

# compiling section
# -- compile shipped libraries
add_subdirectory (thirdparty)

# -- compile source code
add_subdirectory (src)

# testing section
# -- generate configure file for ltp_test
configure_file (
    ${CONFIGURE_DIR}/ltp_all_modules.conf.in
    ${EXECUTABLE_OUTPUT_PATH}/ltp_all_modules.conf)

configure_file (
    ${TOOLS_DIR}/testing/auto-test.sh.in
    ${TOOLS_DIR}/testing/auto-test.sh)

configure_file (
    ${TOOLS_DIR}/train/ltp-model.in
    ${TOOLS_DIR}/train/ltp-model)

# enable test
enable_testing()

# compile testing
add_subdirectory (test)
