# example of configure time generate header

include_directories (${SOURCE_DIR}/)

set (lgdpj_VERSION "0.0.1")

set (lgdpj_SRC collections.cpp
    model.cpp
    featurespace.cpp
    extractor.cpp
    options.cpp
    decoder1o.cpp
    decoder2o.cpp
    parser.cpp)

add_library(parser_static_lib STATIC parser_dll.cpp ${lgdpj_SRC})

set_target_properties (parser_static_lib    PROPERTIES
    OUTPUT_NAME     parser
    PREFIX "lib")

add_library(parser_shared_lib SHARED parser_dll.cpp ${lgdpj_SRC})

set_target_properties (parser_shared_lib    PROPERTIES
    VERSION         ${lgdpj_VERSION}
    OUTPUT_NAME     parser)

# redirect output binary to tools/train
add_executable(lgdpj lgdpj.cpp ${lgdpj_SRC})
set_target_properties (lgdpj
    PROPERTIES
    OUTPUT_NAME lgdpj
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

configure_file (
    parser_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/parser_dll.h)
