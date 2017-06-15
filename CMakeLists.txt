cmake_minimum_required (VERSION 2.8.8)
project ("LTP - Language Technology Platform")

# project attributes section
# -- config cmake modules path
set (CMAKE_MODULE_PATH      ${PROJECT_SOURCE_DIR}/cmake)

if (APPLE)
  add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
  set(CMAKE_CXX_FLAGS "-std=c++0x -Wno-c++11-narrowing")
elseif(UNIX)
  set(CMAKE_CXX_FLAGS "-std=c++0x -fPIC")
elseif(MINGW)
  set(CMAKE_CXX_FLAGS "-std=c++0x")
elseif(MSVC)
  add_definitions(-D_WINDOWS)  # make dynet happy at `dynet/mem.cc(7)`
  add_definitions(-DBOOST_ALL_NO_LIB)  # disable boost auto-linking on windows
  set(CMAKE_CXX_FLAGS "/EHsc")
endif(APPLE)


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
