include_directories (${SOURCE_DIR})

set (splitsnt_VERSION "0.0.1")

add_library (splitsnt_static_lib SplitSentence.h SplitSentence.cpp)

set_target_properties (splitsnt_static_lib  PROPERTIES
    OUTPUT_NAME     splitsnt
    PREFIX "lib")

add_library (splitsnt_shared_lib SHARED SplitSentence.cpp)

set_target_properties (splitsnt_shared_lib    PROPERTIES
    VERSION         ${splitsnt_VERSION}
    OUTPUT_NAME     splitsnt)

configure_file (
    SplitSentence.h
    ${INCLUDE_OUTPUT_PATH}/ltp/SplitSentence.h)
