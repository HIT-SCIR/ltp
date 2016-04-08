include_directories (${SOURCE_DIR}/
    ${THIRDPARTY_DIR}/boost/include
    ${THIRDPARTY_DIR}/eigen-3.2.4/
    )

set (nnsdparser_VERSION "0.0.1")
set (nnsdparser_SRC
    options.cpp
    instance.cpp
    system.cpp
    classifier.cpp
    parser.cpp)

add_library(sdparser_static_lib STATIC sdparser_dll.cpp ${nnsdparser_SRC})
set_target_properties (sdparser_static_lib    PROPERTIES
    OUTPUT_NAME     sdparser)

add_library(sdparser_shared_lib SHARED sdparser_dll.cpp ${nnsdparser_SRC})
set_target_properties (sdparser_shared_lib    PROPERTIES
    VERSION         ${nnsdparser_VERSION}
    OUTPUT_NAME     sdparser)

# redirect output binary to tools/train
add_executable (nnsdparser main.cpp parser_frontend.cpp io.cpp ${nnsdparser_SRC})
target_link_libraries (nnsdparser boost_program_options_static_lib)
set_target_properties (nnsdparser PROPERTIES
    OUTPUT_NAME nnsdparser
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)
configure_file (
    sdparser_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/sdparser_dll.h)
