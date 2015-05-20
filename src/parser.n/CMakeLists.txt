include_directories (${SOURCE_DIR}/
    ${THIRDPARTY_DIR}/boost/include
    ${THIRDPARTY_DIR}/armadillo-5.100.1/include/)

set (nndepparser_VERSION "0.0.1")
set (nndepparser_SRC
    options.cpp
    instance.cpp
    system.cpp
    classifier.cpp
    parser.cpp)

add_library(parser_static_lib STATIC parser_dll.cpp ${nndepparser_SRC})
set_target_properties (parser_static_lib    PROPERTIES
    OUTPUT_NAME     parser)

add_library(parser_shared_lib SHARED parser_dll.cpp ${nndepparser_SRC})
set_target_properties (parser_shared_lib    PROPERTIES
    VERSION         ${nndepparser_VERSION}
    OUTPUT_NAME     parser)

# redirect output binary to tools/train
add_executable (nndepparser main.cpp parser_frontend.cpp io.cpp ${nndepparser_SRC})
target_link_libraries (nndepparser boost_program_options_static_lib)
if (MSVC)
  target_link_libraries (parser_static_lib
      ${THIRDPARTY_DIR}/openblas/lib/libopenblas.dll.a)
  target_link_libraries (parser_shared_lib
      ${THIRDPARTY_DIR}/openblas/lib/libopenblas.dll.a)
  target_link_libraries (nndepparser
      ${THIRDPARTY_DIR}/openblas/lib/libopenblas.dll.a)
else()
  target_link_libraries (parser_static_lib openblas)
  target_link_libraries (parser_shared_lib openblas)
  target_link_libraries (nndepparser openblas)
endif ()
set_target_properties (nndepparser PROPERTIES
    OUTPUT_NAME nndepparser
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)
configure_file (
    parser_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/parser_dll.h)
