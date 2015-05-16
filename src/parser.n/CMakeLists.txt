include_directories (${SOURCE_DIR}/
    ${THIRDPARTY_DIR}/boost/include
    ${THIRDPARTY_DIR}/armadillo-5.100.1/include/)

set (nndeparser_VERSION "0.0.1")
set (nndeparser_SRC
    options.cpp
    instance.cpp
    system.cpp
    classifier.cpp
    parser.cpp)

add_library(parser_static_lib STATIC parser_dll.cpp ${nndeparser_SRC})
set_target_properties (parser_static_lib    PROPERTIES
    OUTPUT_NAME     parser)

add_library(parser_shared_lib SHARED parser_dll.cpp ${nndeparser_SRC})
set_target_properties (parser_shared_lib    PROPERTIES
    VERSION         ${nndeparser_VERSION}
    OUTPUT_NAME     parser)

# redirect output binary to tools/train
add_executable (nndeparser main.cpp parser_frontend.cpp io.cpp ${nndeparser_SRC})
target_link_libraries (nndeparser boost_program_options_static_lib)
if (MSVC)
  target_link_libraries (nndeparser
      ${THIRDPARTY_DIR}/openblas/lib/libopenblas.dll.a)
else()
  target_link_libraries (nndeparser openblas)
endif ()
set_target_properties (nndeparser PROPERTIES
    OUTPUT_NAME nndeparser
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)
configure_file (
    parser_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/parser_dll.h)
