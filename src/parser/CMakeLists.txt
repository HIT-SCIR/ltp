# example of configure time generate header

include_directories (./ 
    ${SOURCE_DIR}/parser 
    ${SOURCE_DIR}/utils 
    ${SOURCE_DIR}/utils/math)

message(STATUS ${SOURCE_DIR})

set (lgdpj_SRC collections.cpp
    model.cpp
    featurespace.cpp
    extractor.cpp
    options.cpp
    decoder1o.cpp
    decoder2o.cpp
    parser.cpp)

add_library(parser parser_dll.cpp ${lgdpj_SRC})

# redirect output binary to tools/train
add_executable(lgdpj lgdpj.cpp ${lgdpj_SRC})
set_target_properties (lgdpj
    PROPERTIES
    OUTPUT_NAME lgdpj
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)


