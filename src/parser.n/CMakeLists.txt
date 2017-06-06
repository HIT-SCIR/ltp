include_directories (${SOURCE_DIR}/
    ${THIRDPARTY_DIR}/boost/include
    ${THIRDPARTY_DIR}/eigen
    )

set (nndepparser_VERSION "0.0.1")
set (nndepparser_SRC
    options.h
    options.cpp
    instance.h
    instance.cpp
    system.h
    system.cpp
    classifier.h
    classifier.cpp
    parser.h
    parser.cpp)

add_library(parser_static_lib STATIC
    parser_dll.h
    parser_dll.cpp
    ${nndepparser_SRC})
set_target_properties (parser_static_lib    PROPERTIES
    OUTPUT_NAME     parser
    PREFIX "lib")

add_library(parser_shared_lib SHARED 
    parser_dll.h
    parser_dll.cpp
    ${nndepparser_SRC})
set_target_properties (parser_shared_lib    PROPERTIES
    VERSION         ${nndepparser_VERSION}
    OUTPUT_NAME     parser)

# redirect output binary to tools/train
add_executable (nndepparser 
    main.cpp
    parser_frontend.h
    parser_frontend.cpp
    io.h
    io.cpp
    ${nndepparser_SRC})
target_link_libraries (nndepparser boost_program_options_static_lib)
set_target_properties (nndepparser PROPERTIES
    OUTPUT_NAME nndepparser
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)
    
configure_file (
    parser_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/parser_dll.h)
