include_directories (./
    ${util_DIR})

set (gparser_SRC
    Alphabet.cpp
    Alphabet.h
    CONLLReader.cpp
    CONLLReader.h
    CONLLWriter.cpp
    CONLLWriter.h
    DepDecoder2O.cpp
    DepDecoder2O.h
    DepDecoder.cpp
    DepDecoder.h
    DepInstance.cpp
    DepInstance.h
    DepParser.cpp
    DepParser.h
    DepPipe2O.cpp
    DepPipe2O.h
    DepPipe.cpp
    DepPipe.h
    DepReader.cpp
    DepReader.h
    DepWriter.cpp
    DepWriter.h
    FeatureVec.cpp
    FeatureVec.h
    gparser_dll.cpp
    gparser_dll.h
    gparser_dll_x.cpp
    KBestParseForest2O.cpp
    KBestParseForest2O.h
    KBestParseForest.cpp
    KBestParseForest.h
    MultiArray.h
    Parameter.cpp
    Parameter.h
    ParseForestItem.cpp
    ParseForestItem.h
    ParserOptions.cpp
    ParserOptions.h)

add_library (gparser_lib ${gparser_SRC})

configure_file (
    gparser_dll.h
    ${INCLUDE_OUTPUT_PATH}/ltp/gparser_dll.h)

target_link_libraries (gparser_lib util)
set_target_properties (gparser_lib PROPERTIES OUTPUT_NAME gparser)

set (gparser_exe_SRC main.cpp)

link_directories (${LIBRARY_OUTPUT_PATH})
add_executable (gparser_exe ${gparser_exe_SRC})
target_link_libraries (gparser_exe gparser)
set_target_properties (gparser_exe
    PROPERTIES
    OUTPUT_NAME gparser
    RUNTIME_OUTPUT_DIRECTORY ${TOOLS_DIR}/train/)

